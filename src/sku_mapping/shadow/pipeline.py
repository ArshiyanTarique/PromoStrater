"""Observational shadow sidecar with hard production-state isolation."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import pickle
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from rapidfuzz import fuzz

from sku_mapping.agreement.policy import evaluate_candidate_agreement
from sku_mapping.config import (
    EmbeddingConfig,
    PipelineConfig,
    ShadowModeConfig,
    load_config,
)
from sku_mapping.constants import FEATURE_GENERATOR_VERSION, MODEL_FEATURE_COLUMNS
from sku_mapping.data.offer_identity import canonical_offer_identity
from sku_mapping.data.preprocessing import preprocess_product_master
from sku_mapping.embedding.scorer import (
    score_candidate_frame_non_blocking,
)
from sku_mapping.embedding.retrieval import (
    EmbeddingRetrievalResult,
    retrieve_embedding_candidates,
)
from sku_mapping.features.commercial_attributes import (
    attributes_json,
    compare_commercial_attributes,
    parse_master_attributes,
    parse_source_attributes,
)
from sku_mapping.features.feature_generator import build_feature_vector
from sku_mapping.features.discriminative_features import build_extra_features
from sku_mapping.features.rank_features import add_rank_features
from sku_mapping.features.measurement_features import (
    pack_is_compatible,
    pack_structure_agrees,
)
from sku_mapping.features.semantic_features import _protein_set
from sku_mapping.failure_diagnostics import capture_exception_details
from sku_mapping.llm_review.reviewer import (
    review_llm_routes_non_blocking,
)
from sku_mapping.matching.candidate_generator import (
    CandidateGenerator,
    CandidateMatch,
)
from sku_mapping.paths import DEFAULT_CONFIG_PATH
from sku_mapping.shadow.challenge import challenge_manifest_template
from sku_mapping.shadow.monitoring import build_shadow_monitoring_report
from sku_mapping.shadow.predictor import (
    RegisteredShadowPackage,
    ShadowPredictor,
    load_registered_shadow_package,
)
from sku_mapping.shadow.review import create_human_review_view
from sku_mapping.shadow.sampling import sample_offers_for_review

LOGGER = logging.getLogger(__name__)
InferenceProgressCallback = Callable[
    [str, int | None, int | None, str], None
]


@dataclass(frozen=True)
class ShadowRunResult:
    """Strict shadow run outcome and separate artifact paths."""

    status: str
    shadow_run_id: str | None
    prediction_rows: int
    offer_groups: int
    failed_shadow_predictions: int
    output_paths: dict[str, Path]
    error: str | None = None
    original_exception: dict[str, Any] | None = None
    exception: BaseException | None = None


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint exact in-memory structure, values, and nested objects."""
    return hashlib.sha256(
        pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # Keep artifact verification reliable on memory-constrained dashboard
        # hosts; hashing does not benefit materially from a 1 MiB allocation.
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


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


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def stable_offer_group_id(row: pd.Series, position: int) -> str:
    """Return the shared deterministic offer identity used by inference."""
    return canonical_offer_identity(row, position)


_stable_offer_group_id = stable_offer_group_id


def _source_row_identifier(row: pd.Series, position: int) -> str:
    for column in ("offerid", "record_id", "source_row_identifier"):
        value = row.get(column)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip()
    return f"shadow-source-row-{position}"


def _prepared_master(master: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Itemcode",
        "Itemname",
        "match_text",
        "master_measures",
        "master_measures_detailed",
        "category",
    }
    if required.issubset(master.columns):
        return master.copy(deep=True)
    return preprocess_product_master(master)


def _production_accepted(value: object, tier: object) -> bool:
    return str(value) in {"AUTO_MATCH", "ACCEPTED"} or str(tier) == "high (ml)"


def _retrieved_candidate(
    offer: pd.Series,
    master: pd.Series,
    *,
    rank: int,
) -> CandidateMatch:
    """Create diagnostics for an embedding-only union candidate."""
    text_score = (
        fuzz.token_sort_ratio(offer["match_text"], master["match_text"])
        + fuzz.token_set_ratio(offer["match_text"], master["match_text"])
    ) / 2.0
    pack_status = pack_is_compatible(
        offer["offer_measures"], master["master_measures"]
    )
    adjusted = text_score + (
        4.0 if pack_status is True else -3.0 if pack_status is None else 0.0
    )
    return CandidateMatch(
        itemcode=str(master["Itemcode"]),
        itemname=str(master["Itemname"]),
        text_score=round(float(text_score), 2),
        adjusted_score=round(float(adjusted), 2),
        margin=0.0,
        raw_margin=0.0,
        pack_status=pack_status,
        pack_structure_status=pack_structure_agrees(
            offer["offer_measures_detailed"],
            master["master_measures_detailed"],
        ),
        category=str(offer["category"]),
        candidate_rank=rank,
        master_match_text=str(master["match_text"]),
        master_measures=tuple(master["master_measures"]),
    )


@dataclass(frozen=True)
class _MasterContext:
    """Master-side lookups and the candidate generator, built once per run.

    Every one of these is derived purely from the product master, so rebuilding
    them for each streaming chunk would repeat the master parse and the
    generator's per-category pool construction for no benefit.
    """

    master: pd.DataFrame
    master_lookup: dict[str, pd.Series]
    master_commercial: dict[str, Any]
    generator: CandidateGenerator


def _build_master_context(product_master: pd.DataFrame) -> _MasterContext:
    """Prepare the master-side state shared by every chunk."""
    master = _prepared_master(product_master)
    master_lookup = {
        str(row["Itemcode"]): row for _, row in master.iterrows()
    }
    master_commercial = {
        itemcode: parse_master_attributes(row)
        for itemcode, row in master_lookup.items()
    }
    return _MasterContext(
        master=master,
        master_lookup=master_lookup,
        master_commercial=master_commercial,
        generator=CandidateGenerator(master),
    )


def _prepare_own_offers(production_rows: pd.DataFrame) -> pd.DataFrame:
    """Filter to own-brand offers and assign identities from global positions.

    This must run exactly once over the whole frame, never per chunk.
    ``canonical_offer_identity`` and ``_source_row_identifier`` both fall back
    to the row's integer position whenever ``offerid`` is missing, so assigning
    identities inside a chunk would mint different ids for the same offer and
    silently change every downstream artifact.
    """
    own = production_rows.copy(deep=True)
    if "is_own" in own:
        own = own[own["is_own"].fillna(False).astype(bool)].copy()
    own.reset_index(drop=True, inplace=True)
    if own.empty:
        return own
    own["offer_group_id"] = [
        (
            str(row.get("offer_group_id"))
            if int(row.get("entity_count", 1) or 1) > 1
            else _stable_offer_group_id(row, position)
        )
        for position, (_, row) in enumerate(own.iterrows())
    ]
    own["source_row_identifier"] = [
        _source_row_identifier(row, position)
        for position, (_, row) in enumerate(own.iterrows())
    ]
    return own


def _build_candidate_features_for_slice(
    own: pd.DataFrame,
    master_context: _MasterContext,
    *,
    top_k: int,
    retain_all_candidates: bool,
    shadow_run_id: str,
    timestamp: str,
    registered: RegisteredShadowPackage,
    embedding_config: EmbeddingConfig,
    progress: InferenceProgressCallback | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> tuple[
    pd.DataFrame,
    int,
    EmbeddingRetrievalResult | None,
]:
    """Build candidate rows for one already-prepared slice of own-brand offers.

    ``own`` must come from :func:`_prepare_own_offers` and be re-indexed from
    zero: positions here are slice-local by design, because identities were
    already assigned globally. ``progress_offset``/``progress_total`` exist only
    so a chunked caller can report cumulative progress; they never influence
    the rows produced.
    """
    if own.empty:
        return pd.DataFrame(), 0, None

    master = master_context.master
    master_lookup = master_context.master_lookup
    master_commercial = master_context.master_commercial
    reported_total = (
        len(own) if progress_total is None else progress_total
    )
    source_commercial = {
        position: parse_source_attributes(own.iloc[position])
        for position in range(len(own))
    }
    ranked = master_context.generator.generate_candidates_batch(
        own,
        top_k=top_k,
        progress=(
            (
                lambda completed, total: progress(
                    "candidate_generation",
                    progress_offset + completed,
                    reported_total,
                    f"Generated candidates for "
                    f"{progress_offset + completed:,} of "
                    f"{reported_total:,} own-brand offers",
                )
            )
            if progress is not None
            else None
        ),
    )
    retrieval_result = retrieve_embedding_candidates(
        own,
        master,
        config=embedding_config,
    )
    retrieval_similarity: dict[tuple[int, str], float] = {}
    if retain_all_candidates:
        for offer_position, hits in enumerate(retrieval_result.hits):
            existing = {
                candidate.itemcode for candidate in ranked[offer_position]
            }
            for hit in hits:
                retrieval_similarity[
                    (offer_position, hit.master_itemcode)
                ] = hit.similarity
                if hit.master_itemcode in existing:
                    continue
                master_row = master_lookup.get(hit.master_itemcode)
                if master_row is None:
                    continue
                ranked[offer_position].append(
                    _retrieved_candidate(
                        own.iloc[offer_position],
                        master_row,
                        rank=len(ranked[offer_position]) + 1,
                    )
                )
                existing.add(hit.master_itemcode)
                if len(ranked[offer_position]) >= top_k * 2:
                    break
    rows: list[dict[str, Any]] = []
    failures = 0
    feature_update_interval = max(1, len(ranked) // 100)
    for offer_position, candidates in enumerate(ranked):
        offer = own.iloc[offer_position]
        selected_candidates = (
            candidates if retain_all_candidates else candidates[:1]
        )
        for candidate in selected_candidates:
            if candidate.candidate_rank < 1 or candidate.itemcode not in master_lookup:
                failures += 1
                continue
            master_row = master_lookup[candidate.itemcode]
            try:
                feature_values = build_feature_vector(offer, master_row)
                comparison = compare_commercial_attributes(
                    source_commercial[offer_position],
                    master_commercial[candidate.itemcode],
                )
            except Exception:
                failures += 1
                LOGGER.exception(
                    "Shared feature generation failed for shadow candidate"
                )
                continue
            feature_row = {
                feature: feature_values[feature]
                for feature in MODEL_FEATURE_COLUMNS
            }
            # Add extra discriminative features if the model uses them.
            if registered.package.get("requires_group_features"):
                offer_text_for_extra = " ".join(filter(None, [
                    str(offer.get("Offer Name", "")),
                    str(offer.get("Product", "")),
                ]))
                master_text_for_extra = " ".join(filter(None, [
                    str(master_row.get("Itemname", "")),
                    str(master_row.get("Item-Spec", "")),
                ]))
                feature_row.update(build_extra_features(offer_text_for_extra, master_text_for_extra))
            missing_flags = {
                f"missing_{feature}": bool(pd.isna(feature_row.get(feature)))
                for feature in feature_row
            }
            offer_text = str(offer.get("Offer Name", ""))
            proteins = sorted(
                _protein_set(
                    " ".join(
                        [
                            offer_text,
                            str(offer.get("Product", "")),
                            str(offer.get("Variant", "")),
                        ]
                    )
                )
            )
            production_decision = offer.get(
                "ml_decision",
                offer.get("production_decision", "UNKNOWN"),
            )
            production_tier = offer.get(
                "confidence_tier",
                offer.get("production_tier", "UNKNOWN"),
            )
            pack_conflict = bool(
                candidate.pack_status is False
                or candidate.pack_structure_status is False
                or feature_row["pack_format_match"] == 0
            )
            rows.append(
                {
                    "source_commercial_attributes": attributes_json(
                        source_commercial[offer_position]
                    ),
                    "master_commercial_attributes": attributes_json(
                        master_commercial[candidate.itemcode]
                    ),
                    **comparison.to_record(),
                    "shadow_run_id": shadow_run_id,
                    "timestamp": timestamp,
                    "model_id": registered.package["model_id"],
                    "model_package_sha256": registered.package_sha256,
                    "feature_generator_version": FEATURE_GENERATOR_VERSION,
                    "source_dataset": str(
                        offer.get(
                            "source_dataset", "CLICKFLYER_PRODUCTION"
                        )
                    ),
                    "source_row_identifier": offer[
                        "source_row_identifier"
                    ],
                    "source_offer_id": str(
                        offer.get(
                            "source_offer_id", offer["offer_group_id"]
                        )
                    ),
                    "source_offer_text": str(
                        offer.get("source_offer_text", offer_text)
                    ),
                    "entity_id": str(
                        offer.get("entity_id", offer["offer_group_id"])
                    ),
                    "entity_index": int(
                        offer.get("entity_index", 1) or 1
                    ),
                    "entity_count": int(
                        offer.get("entity_count", 1) or 1
                    ),
                    "entity_text": str(
                        offer.get("entity_text", offer_text)
                    ),
                    "entity_type": str(
                        offer.get("entity_type", "SINGLE_PRODUCT")
                    ),
                    "conjunction_type": str(
                        offer.get("conjunction_type", "SINGLE")
                    ),
                    "entity_protein": str(
                        offer.get("entity_protein", "")
                    ),
                    "entity_product_family": str(
                        offer.get("entity_product_family", "")
                    ),
                    "entity_retail_weight_g": offer.get(
                        "entity_retail_weight_g", pd.NA
                    ),
                    "attribute_inheritance_flags": str(
                        offer.get("attribute_inheritance_flags", "")
                    ),
                    "entity_parse_confidence": offer.get(
                        "entity_parse_confidence", pd.NA
                    ),
                    "offer_group_id": offer["offer_group_id"],
                    "offer_text": offer_text,
                    "offer_brand": str(offer.get("Brand Name", "")),
                    "offer_product": str(offer.get("Product", "")),
                    "offer_variant": str(offer.get("Variant", "")),
                    "offer_base_packsize": str(
                        offer.get("Base Packsize", "")
                    ),
                    "candidate_rank": int(candidate.candidate_rank),
                    "candidate_retrieval_source": (
                        "FUZZY_AND_EMBEDDING"
                        if candidate.candidate_rank <= top_k
                        and (
                            offer_position,
                            candidate.itemcode,
                        )
                        in retrieval_similarity
                        else (
                            "FUZZY"
                            if candidate.candidate_rank <= top_k
                            else "EMBEDDING_EXPANSION"
                        )
                    ),
                    "retrieval_embedding_similarity": (
                        retrieval_similarity.get(
                            (offer_position, candidate.itemcode),
                            pd.NA,
                        )
                    ),
                    "master_itemcode": candidate.itemcode,
                    "master_brand": str(
                        master_row.get("Brand Name", "Al Kabeer")
                    ),
                    "master_item_description": str(
                        master_row.get("Itemname", "")
                    ),
                    "master_item_family": str(
                        master_row.get("Item-Cat-4", "")
                    ),
                    "master_item_category": str(
                        master_row.get("Item-Cat-2", "")
                    ),
                    "master_item_long_description": str(
                        master_row.get("Item Description", "")
                    ),
                    "master_item_spec": str(
                        master_row.get("Item-Spec", "")
                    ),
                    "candidate_text_score": candidate.text_score,
                    "candidate_adjusted_score": candidate.adjusted_score,
                    "candidate_margin": candidate.margin,
                    "candidate_raw_margin": candidate.raw_margin,
                    "candidate_category": candidate.category,
                    "candidate_pack_status": candidate.pack_status,
                    "candidate_pack_structure_status": (
                        candidate.pack_structure_status
                    ),
                    "existing_production_decision": str(production_decision),
                    "existing_production_tier": str(production_tier),
                    "production_ml_mode": str(
                        offer.get("assisted_mode", "shadow")
                    ),
                    "assisted_decision": str(
                        offer.get("assisted_decision", "")
                    ),
                    "assisted_decision_reason": str(
                        offer.get("assisted_decision_reason", "")
                    ),
                    "assisted_safety_override_applied": bool(
                        offer.get(
                            "assisted_safety_override_applied", False
                        )
                    ),
                    "assisted_conflict_flags": str(
                        offer.get("assisted_conflict_flags", "")
                    ),
                    "assisted_auto_accept_threshold": offer.get(
                        "assisted_auto_accept_threshold", pd.NA
                    ),
                    "assisted_threshold_source": str(
                        offer.get("assisted_threshold_source", "")
                    ),
                    "assisted_production_threshold_approved": bool(
                        offer.get(
                            "assisted_production_threshold_approved", False
                        )
                    ),
                    "current_production_match_itemcode": str(
                        offer.get("matched_itemcode", "")
                    ),
                    "current_production_suggested_itemcode": str(
                        offer.get("suggested_itemcode", "")
                    ),
                    "diagnostic_high_score_threshold": float(
                        registered.package["auto_match_threshold"]
                    ),
                    "diagnostic_review_threshold": float(
                        registered.package["manual_review_threshold"]
                    ),
                    "feature_missingness_count": int(
                        sum(missing_flags.values())
                    ),
                    "product_family": str(
                        offer.get(
                            "product_family",
                            offer.get("product_class_offer", ""),
                        )
                    ),
                    "protein_classification": "|".join(proteins)
                    if proteins
                    else "<unspecified>",
                    "pack_conflict": pack_conflict,
                    "provenance": str(
                        offer.get("label_provenance", "")
                    ),
                    "human_review_status": "UNREVIEWED",
                    "human_label": "",
                    "reviewer_notes": "",
                    "review_timestamp": "",
                    **feature_row,
                    **missing_flags,
                }
            )
        completed_offers = offer_position + 1
        if progress is not None and (
            completed_offers == len(ranked)
            or completed_offers % feature_update_interval == 0
        ):
            progress(
                "feature_generation",
                progress_offset + completed_offers,
                reported_total,
                f"Built features for "
                f"{progress_offset + completed_offers:,} of "
                f"{reported_total:,} own-brand offers",
            )
    return pd.DataFrame(rows), failures, retrieval_result


def _build_candidate_features(
    production_rows: pd.DataFrame,
    product_master: pd.DataFrame,
    *,
    top_k: int,
    retain_all_candidates: bool,
    shadow_run_id: str,
    timestamp: str,
    registered: RegisteredShadowPackage,
    embedding_config: EmbeddingConfig,
    progress: InferenceProgressCallback | None = None,
) -> tuple[
    pd.DataFrame,
    int,
    EmbeddingRetrievalResult | None,
]:
    """Build every candidate row in one pass (the legacy, unchunked path)."""
    own = _prepare_own_offers(production_rows)
    if own.empty:
        return pd.DataFrame(), 0, None
    return _build_candidate_features_for_slice(
        own,
        _build_master_context(product_master),
        top_k=top_k,
        retain_all_candidates=retain_all_candidates,
        shadow_run_id=shadow_run_id,
        timestamp=timestamp,
        registered=registered,
        embedding_config=embedding_config,
        progress=progress,
    )


#: Scored chunks are spilled here while a streaming run is in flight.
_SPOOL_DIRECTORY_NAME = "_chunk_spool"

#: Heavy candidate columns that no post-chunk global stage reads. Dropping them
#: when the spool is read back keeps sampling, the review view, and monitoring
#: off the long description and JSON-blob columns. Verified against
#: ``shadow/sampling.py``, ``shadow/review.py`` and ``shadow/monitoring.py``:
#: none of the three references any name below.
_GLOBAL_STAGE_EXCLUDED_COLUMNS = (
    "source_commercial_attributes",
    "master_commercial_attributes",
    "master_item_long_description",
    "master_item_spec",
    "master_item_family",
    "master_item_category",
    "offer_base_packsize",
)


@dataclass(frozen=True)
class _AgreementTotals:
    """Agreement rows accumulated across chunks, in chunk order."""

    frame: pd.DataFrame


def _merge_llm_results(results: list[Any]) -> Any:
    """Combine per-chunk LLM review results into one run-level view.

    Provider identity and prompt versions are run-level constants, so they come
    from the first chunk; rows, per-offer results, and counters accumulate. A
    single chunk returns its result unchanged, leaving the legacy path exactly
    as it was.
    """
    if len(results) == 1:
        return results[0]
    first = results[0]
    failed = next(
        (item for item in results if str(item.status) != str(first.status)),
        None,
    )
    combined_results: tuple[Any, ...] = ()
    for item in results:
        combined_results = combined_results + tuple(item.results)
    return replace(
        first,
        status=failed.status if failed is not None else first.status,
        error=(
            getattr(failed, "error", None)
            if failed is not None and getattr(failed, "error", None)
            else getattr(first, "error", None)
        ),
        results=combined_results,
        frame=pd.concat(
            [item.frame for item in results], ignore_index=True
        ),
        offers_routed=sum(int(item.offers_routed or 0) for item in results),
        provider_calls=sum(
            int(item.provider_calls or 0) for item in results
        ),
        cache_hits=sum(int(item.cache_hits or 0) for item in results),
        failures=sum(int(item.failures or 0) for item in results),
    )


@dataclass
class _ScoredChunk:
    """One chunk after prediction, embedding, agreement, and LLM routing."""

    frame: pd.DataFrame
    agreement_frame: pd.DataFrame
    llm_result: Any
    embedding_result: Any
    stage_runtimes: dict[str, float]

    @property
    def embedding_status(self) -> str:
        return str(self.embedding_result.status)


def _plan_chunk_edges(
    own: pd.DataFrame, chunk_size: int
) -> list[tuple[int, int]]:
    """Split own-brand offers into slices that never split a source offer.

    ``finalize_unified_decisions`` aggregates entities by ``source_offer_id``
    and one source row can expand into several offers, so a boundary drawn
    through a group would change that aggregate. Edges are pushed forward to
    the next source-offer change, which can make the final chunk slightly
    larger than requested.
    """
    total = len(own)
    if chunk_size <= 0 or total <= chunk_size:
        return [(0, total)] if total else []
    source_ids = (
        own["source_offer_id"].to_numpy()
        if "source_offer_id" in own.columns
        else None
    )
    edges: list[tuple[int, int]] = []
    start = 0
    while start < total:
        stop = min(start + chunk_size, total)
        if source_ids is not None:
            while stop < total and source_ids[stop] == source_ids[stop - 1]:
                stop += 1
        edges.append((start, stop))
        start = stop
    return edges


def _score_candidate_chunk(
    candidate_frame: pd.DataFrame,
    *,
    predictor: ShadowPredictor,
    config: PipelineConfig,
) -> _ScoredChunk:
    """Apply prediction, embedding, agreement, and LLM routing to one frame.

    Lifted verbatim from the single-pass implementation so that the chunked and
    legacy paths run identical code. Every stage here is per-offer-group
    independent, and chunks are offer-disjoint, so slicing the input does not
    change any result.
    """
    stage_runtimes: dict[str, float] = {}

    stage_started = time.perf_counter()
    pkg_cols = list(predictor.feature_columns)
    requires_rank = predictor.package.get("requires_group_features", False)
    if requires_rank:
        offer_id_col = next(
            (c for c in ("offer_group_id", "offer_id", "source_offer_id")
             if c in candidate_frame.columns),
            None,
        )
        if offer_id_col is None:
            raise ValueError(
                "Ranked model requires offer_group_id/offer_id column in candidate_frame"
            )
        parts = []
        for _, grp in candidate_frame.groupby(offer_id_col, sort=False):
            parts.append(add_rank_features(grp.copy()))
        enriched = pd.concat(parts).reindex(candidate_frame.index)
        feature_frame = enriched.reindex(columns=pkg_cols).fillna(-1.0)
    else:
        feature_frame = candidate_frame.loc[:, MODEL_FEATURE_COLUMNS]
    predictions = predictor.predict(feature_frame)
    candidate_frame = pd.concat(
        [
            candidate_frame.reset_index(drop=True),
            predictions.reset_index(drop=True),
        ],
        axis=1,
    )
    stage_runtimes["lightgbm_scoring"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    embedding_result = score_candidate_frame_non_blocking(
        candidate_frame,
        config=config.embedding,
    )
    embedding_scores = embedding_result.scores
    if embedding_scores.empty and len(candidate_frame):
        embedding_scores = pd.DataFrame(
            {
                "embedding_status": embedding_result.status,
                "embedding_model_id": (
                    f"{config.embedding.backend}:"
                    f"{config.embedding.model_name}"
                ),
                "embedding_model_version": (
                    config.embedding.model_version or "NOT_LOADED"
                ),
                "offer_text_used": "",
                "candidate_text_used": "",
                "embedding_similarity": float("nan"),
                "embedding_rank": pd.array(
                    [pd.NA] * len(candidate_frame), dtype="Int64"
                ),
                "embedding_top_candidate": False,
                "embedding_failure_reason": (
                    "DISABLED_BY_CONFIGURATION"
                    if embedding_result.status == "DISABLED"
                    else embedding_result.error or "EMBEDDING_UNAVAILABLE"
                ),
            },
            index=candidate_frame.index,
        )
    candidate_frame = pd.concat(
        [
            candidate_frame,
            embedding_scores.reindex(candidate_frame.index),
        ],
        axis=1,
    )
    stage_runtimes["embedding_scoring"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    agreement_result = evaluate_candidate_agreement(
        candidate_frame,
        config=config.agreement,
    )
    agreement_candidate_columns = agreement_result.frame.rename(
        columns={
            "embedding_top_candidate": (
                "agreement_embedding_top_candidate"
            ),
            "embedding_similarity": "agreement_embedding_similarity",
            "embedding_rank": "agreement_embedding_rank",
        }
    )
    candidate_frame = candidate_frame.merge(
        agreement_candidate_columns,
        left_on="offer_group_id",
        right_on="offer_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    stage_runtimes["agreement_policy"] = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    llm_review_result = review_llm_routes_non_blocking(
        candidate_frame,
        agreement_result.frame,
        config=config.llm_review,
    )
    candidate_frame = candidate_frame.merge(
        llm_review_result.frame,
        left_on="offer_group_id",
        right_on="offer_id",
        how="left",
        sort=False,
        validate="many_to_one",
        suffixes=("", "_llm"),
    )
    stage_runtimes["llm_review"] = time.perf_counter() - stage_started

    production_accepted = [
        _production_accepted(decision, tier)
        for decision, tier in zip(
            candidate_frame["existing_production_decision"],
            candidate_frame["existing_production_tier"],
        )
    ]
    candidate_frame["production_shadow_disagreement"] = (
        pd.Series(production_accepted, index=candidate_frame.index)
        .ne(
            candidate_frame["shadow_decision_bucket"].eq(
                "SHADOW_HIGH_SCORE"
            )
        )
        .astype(bool)
    )
    forbidden_shadow_actions = set(
        candidate_frame["shadow_decision_bucket"]
    ) - {
        "SHADOW_HIGH_SCORE",
        "SHADOW_REVIEW",
        "SHADOW_LOW_SCORE",
    }
    if forbidden_shadow_actions:
        raise AssertionError(
            f"Shadow output contains production actions: "
            f"{sorted(forbidden_shadow_actions)}"
        )
    return _ScoredChunk(
        frame=candidate_frame,
        agreement_frame=agreement_result.frame,
        llm_result=llm_review_result,
        embedding_result=embedding_result,
        stage_runtimes=stage_runtimes,
    )


def _merge_embedding_results(results: list[Any]) -> Any:
    """Combine per-chunk embedding results into one run-level view.

    Backend identity, status, and device are properties of the configured
    model and are therefore taken from the first chunk; only the per-run
    tallies accumulate. With a single chunk this returns that chunk's result
    unchanged, so the legacy path is unaffected.
    """
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    first = results[0]
    failed = next(
        (item for item in results if str(item.status) != str(first.status)),
        None,
    )
    return replace(
        first,
        status=failed.status if failed is not None else first.status,
        error=(
            failed.error
            if failed is not None and failed.error
            else first.error
        ),
        candidates_scored=sum(
            int(item.candidates_scored or 0) for item in results
        ),
        failures=sum(int(item.failures or 0) for item in results),
        runtime_seconds=sum(
            float(item.runtime_seconds or 0.0) for item in results
        ),
    )


def _stream_parquet(sources: list[Path], destination: Path) -> None:
    """Concatenate spooled chunks into one parquet without loading them all."""
    import pyarrow.parquet as pq

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    writer = None
    try:
        for source in sources:
            table = pq.read_table(source)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema)
            writer.write_table(table)
            del table
        if writer is not None:
            writer.close()
            writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def _stream_csv(sources: list[Path], destination: Path) -> None:
    """Concatenate spooled chunks into one CSV, one chunk in memory at a time.

    Matches :func:`_atomic_csv` byte for byte: the handle is opened once with
    ``utf-8-sig`` so the BOM is written exactly once, and only the first chunk
    contributes a header.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with open(
            temporary, "w", encoding="utf-8-sig", newline=""
        ) as handle:
            for position, source in enumerate(sources):
                frame = pd.read_parquet(source)
                frame.to_csv(handle, index=False, header=position == 0)
                del frame
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _read_spool_projection(sources: list[Path]) -> pd.DataFrame:
    """Read spooled chunks back with the heavy candidate columns dropped."""
    import pyarrow.parquet as pq

    if not sources:
        return pd.DataFrame()
    # Each chunk is projected against its OWN schema. Reading every chunk with
    # the first chunk's column list assumed all chunks carry the same columns,
    # which held only while no chunk could differ: with routing inert nothing
    # was ever reviewed, so the LLM columns were absent from all of them
    # equally. Once one chunk contains a reviewed offer and another does not,
    # their schemas genuinely differ and demanding chunk 0's columns from a
    # chunk that lacks them raises rather than yielding a null column.
    schemas = [pq.read_schema(source).names for source in sources]
    keeps = [
        [name for name in names if name not in _GLOBAL_STAGE_EXCLUDED_COLUMNS]
        for names in schemas
    ]
    # Union in first-seen order, so the column order a caller sees does not
    # depend on which chunk happened to introduce a column. A single-pass run
    # produces one frame carrying every column, and concat below fills a
    # column a chunk never had with NaN - the same value a single-pass row
    # that was never reviewed already carries.
    ordered: list[str] = []
    seen: set[str] = set()
    for keep in keeps:
        for name in keep:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
    frames = [
        pd.read_parquet(source, columns=keep)
        for source, keep in zip(sources, keeps)
    ]
    combined = pd.concat(frames, ignore_index=True)
    frames.clear()
    return combined.reindex(columns=ordered)


def run_shadow_observation(
    production_rows: pd.DataFrame,
    product_master: pd.DataFrame,
    *,
    config: PipelineConfig,
    model_directory: str | Path | None = None,
    shadow_run_id: str | None = None,
    progress: InferenceProgressCallback | None = None,
) -> ShadowRunResult:
    """Run strict shadow inference without mutating production-owned objects."""
    if not config.shadow_mode.enabled:
        return ShadowRunResult(
            status="DISABLED",
            shadow_run_id=None,
            prediction_rows=0,
            offer_groups=0,
            failed_shadow_predictions=0,
            output_paths={},
        )
    run_started = time.perf_counter()
    stage_runtimes: dict[str, float] = {}
    before_production = production_rows.copy(deep=True)
    before_master = product_master.copy(deep=True)
    production_fingerprint = _frame_fingerprint(production_rows)
    master_fingerprint = _frame_fingerprint(product_master)

    shadow = config.shadow_mode
    effective_model_directory = (
        Path(model_directory)
        if model_directory is not None
        else (
            shadow.package_reference.parent
            if shadow.package_reference is not None
            else shadow.registry_path.parent / "registry"
        )
    )
    registered = load_registered_shadow_package(
        registry_path=shadow.registry_path,
        model_directory=effective_model_directory,
        model_id=shadow.model_id,
        package_reference=shadow.package_reference,
        require_package_status=shadow.require_package_status,
    )
    predictor = ShadowPredictor(registered)
    now = datetime.now(timezone.utc)
    timestamp = now.isoformat()
    effective_run_id = shadow_run_id or (
        f"shadow-{now.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{hashlib.sha256((predictor.model_id + production_fingerprint).encode('utf-8')).hexdigest()[:12]}"
    )
    run_directory = shadow.output_directory / effective_run_id
    if run_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing shadow run: {run_directory}"
        )

    own_offers = _prepare_own_offers(production_rows.copy(deep=True))
    chunk_edges = _plan_chunk_edges(own_offers, shadow.chunk_size)
    spool_directory = run_directory / _SPOOL_DIRECTORY_NAME
    spool_paths: list[Path] = []
    streaming = shadow.chunk_size > 0 and len(chunk_edges) > 1

    master_context = _build_master_context(product_master.copy(deep=True))
    feature_failures = 0
    embedding_retrieval_result: EmbeddingRetrievalResult | None = None
    agreement_frames: list[pd.DataFrame] = []
    llm_results: list[Any] = []
    llm_provider_calls = 0
    embedding_results: list[Any] = []
    scored_frames: list[pd.DataFrame] = []
    total_own_offers = len(own_offers)
    total_candidate_rows = 0

    for chunk_index, (start, stop) in enumerate(chunk_edges):
        stage_started = time.perf_counter()
        chunk_offers = own_offers.iloc[start:stop].reset_index(drop=True)
        (
            chunk_frame,
            chunk_failures,
            chunk_retrieval,
        ) = _build_candidate_features_for_slice(
            chunk_offers,
            master_context,
            top_k=shadow.top_k,
            retain_all_candidates=shadow.retain_all_candidates,
            shadow_run_id=effective_run_id,
            timestamp=timestamp,
            registered=registered,
            embedding_config=config.embedding,
            progress=progress,
            progress_offset=start,
            progress_total=total_own_offers,
        )
        stage_runtimes["candidate_generation_and_features"] = (
            stage_runtimes.get("candidate_generation_and_features", 0.0)
            + (time.perf_counter() - stage_started)
        )
        feature_failures += chunk_failures
        if embedding_retrieval_result is None:
            embedding_retrieval_result = chunk_retrieval
        if chunk_frame.empty:
            del chunk_frame, chunk_offers
            continue

        # Emit a pre-scoring tick so the bar moves immediately when
        # LightGBM scoring starts, not only after the chunk completes.
        if progress is not None:
            progress(
                "lightgbm",
                start,
                total_own_offers,
                f"Scoring {stop - start:,} offers (chunk {chunk_index + 1} of {len(chunk_edges)})",
            )

        scored = _score_candidate_chunk(
            chunk_frame,
            predictor=predictor,
            config=config,
        )
        for stage_name, elapsed in scored.stage_runtimes.items():
            stage_runtimes[stage_name] = (
                stage_runtimes.get(stage_name, 0.0) + elapsed
            )
        agreement_frames.append(scored.agreement_frame)
        llm_results.append(scored.llm_result)
        llm_provider_calls += int(scored.llm_result.provider_calls or 0)
        embedding_results.append(scored.embedding_result)
        total_candidate_rows += len(scored.frame)

        # Progress is reported cumulatively: the dashboard tracker keeps the
        # maximum fraction per stage, so per-chunk counts would stall the bar.
        if progress is not None:
            progress(
                "lightgbm",
                stop,
                total_own_offers,
                f"Scored {total_candidate_rows:,} candidate pairs",
            )
            progress(
                "embedding",
                stop,
                total_own_offers,
                (
                    f"Embedding status: {scored.embedding_status}; "
                    f"{total_candidate_rows:,} candidate pairs checked"
                ),
            )
            progress(
                "agreement",
                stop,
                total_own_offers,
                f"Applied agreement policy to {stop:,} offers",
            )
            progress(
                "llm_review",
                stop,
                total_own_offers,
                f"Completed {llm_provider_calls:,} LLM provider calls",
            )

        if streaming:
            spool_directory.mkdir(parents=True, exist_ok=True)
            spool_path = (
                spool_directory / f"chunk_{chunk_index:05d}.parquet"
            )
            scored.frame.to_parquet(spool_path, index=False)
            spool_paths.append(spool_path)
            if progress is not None:
                progress(
                    "feature_generation",
                    stop,
                    total_own_offers,
                    f"Completed chunk {chunk_index + 1:,} of "
                    f"{len(chunk_edges):,} "
                    f"({stop:,} of {total_own_offers:,} offers)",
                )
        else:
            scored_frames.append(scored.frame)

        # Release the chunk before the next one is built, so peak memory is
        # bounded by one chunk rather than by the whole run.
        del scored, chunk_frame, chunk_offers
        gc.collect()

    del own_offers, master_context
    gc.collect()

    if streaming:
        candidate_frame = _read_spool_projection(spool_paths)
    elif scored_frames:
        candidate_frame = (
            scored_frames[0]
            if len(scored_frames) == 1
            else pd.concat(scored_frames, ignore_index=True)
        )
    else:
        candidate_frame = pd.DataFrame()
    scored_frames.clear()

    if candidate_frame.empty:
        raise ValueError("Shadow run produced no evaluable candidate rows")

    agreement_frame = (
        agreement_frames[0]
        if len(agreement_frames) == 1
        else pd.concat(agreement_frames, ignore_index=True)
    )
    agreement_frames.clear()
    embedding_result = _merge_embedding_results(embedding_results)
    llm_review_result = _merge_llm_results(llm_results)
    llm_results.clear()
    agreement_result = _AgreementTotals(frame=agreement_frame)

    sample = sample_offers_for_review(
        candidate_frame,
        counts_by_stratum=dict(shadow.sampling_counts),
        random_seed=config.runtime.random_seed,
        diagnostic_threshold=predictor.high_score_threshold,
    )
    review_view = create_human_review_view(
        sample.offers,
        candidate_frame,
        top_k=shadow.top_k,
    )
    monitoring = build_shadow_monitoring_report(
        candidate_frame,
        model_id=predictor.model_id,
        package_sha256=registered.package_sha256,
        failed_shadow_predictions=feature_failures,
        package_validation_failures=0,
        llm_review_summary={
            "enabled": config.llm_review.enabled,
            "offers_routed": llm_review_result.offers_routed,
            "provider_calls": llm_review_result.provider_calls,
            "cache_hits": llm_review_result.cache_hits,
        },
    )
    output_paths = {
        "shadow_predictions_parquet": run_directory
        / "shadow_predictions.parquet",
        "shadow_predictions_csv": run_directory / "shadow_predictions.csv",
        "human_review_template": run_directory
        / "human_review_template.csv",
        "sampling_report": run_directory / "sampling_report.json",
        "review_intake_audit": run_directory
        / "review_intake_audit.json",
        "monitoring_report": run_directory / "monitoring_report.json",
        "agreement_results": run_directory / "agreement_results.csv",
        "llm_review_results": run_directory / "llm_review_results.csv",
        "run_manifest": run_directory / "shadow_run_manifest.json",
        "challenge_manifest_template": shadow.challenge_set_directory
        / "challenge_set_manifest_template.json",
    }
    if streaming:
        # Assemble the candidate artifacts straight from the spool so the full
        # frame is never materialised with every column at once.
        _stream_parquet(
            spool_paths, output_paths["shadow_predictions_parquet"]
        )
        _stream_csv(spool_paths, output_paths["shadow_predictions_csv"])
    else:
        _atomic_parquet(
            candidate_frame, output_paths["shadow_predictions_parquet"]
        )
        _atomic_csv(
            candidate_frame, output_paths["shadow_predictions_csv"]
        )
    _atomic_csv(review_view, output_paths["human_review_template"])
    _atomic_json(sample.report, output_paths["sampling_report"])
    _atomic_json(
        {
            "status": "NO_COMPLETED_REVIEW_SUBMITTED",
            "rows_staged": 0,
            "training_data_updated": False,
            "instructions": "See docs/HUMAN_REVIEW_GUIDE.md.",
        },
        output_paths["review_intake_audit"],
    )
    _atomic_json(monitoring, output_paths["monitoring_report"])
    _atomic_csv(
        agreement_result.frame, output_paths["agreement_results"]
    )
    _atomic_csv(
        llm_review_result.frame, output_paths["llm_review_results"]
    )
    if streaming:
        # The spool only exists to bound peak memory during the run; the
        # assembled artifacts above are the durable outputs.
        for spool_path in spool_paths:
            spool_path.unlink(missing_ok=True)
        spool_directory.rmdir()

    template_path = output_paths["challenge_manifest_template"]
    if template_path.exists():
        existing = json.loads(template_path.read_text(encoding="utf-8"))
        if existing != challenge_manifest_template():
            raise FileExistsError(
                "Existing challenge-set manifest template differs; refusing "
                "to overwrite it"
            )
    else:
        _atomic_json(challenge_manifest_template(), template_path)

    pd.testing.assert_frame_equal(
        production_rows, before_production, check_exact=True
    )
    pd.testing.assert_frame_equal(
        product_master, before_master, check_exact=True
    )
    if _frame_fingerprint(production_rows) != production_fingerprint:
        raise AssertionError("Shadow inference altered production rows")
    if _frame_fingerprint(product_master) != master_fingerprint:
        raise AssertionError("Shadow inference altered Product Master rows")

    manifest = {
        "shadow_run_id": effective_run_id,
        "status": "COMPLETED_OBSERVATIONAL_ONLY",
        "timestamp": timestamp,
        "model_id": predictor.model_id,
        "model_package_filename": registered.package_path.name,
        "model_package_sha256": registered.package_sha256,
        "deployment_status": "SHADOW_MODE_ONLY",
        "automatic_production_matching_approved": False,
        "approved_auto_match_threshold": None,
        "shadow_decision_terminology": [
            "SHADOW_HIGH_SCORE",
            "SHADOW_REVIEW",
            "SHADOW_LOW_SCORE",
        ],
        "prediction_rows": int(len(candidate_frame)),
        "offer_groups": int(candidate_frame["offer_group_id"].nunique()),
        "failed_shadow_predictions": int(feature_failures),
        "embedding_scoring": {
            "enabled": config.embedding.enabled,
            "requested": embedding_result.requested,
            "available": embedding_result.available,
            "used": embedding_result.used,
            "status": embedding_result.status,
            "backend": config.embedding.backend,
            "model_name": config.embedding.model_name,
            "model_version": (
                config.embedding.model_version or "UNRESOLVED"
            ),
            "device": embedding_result.device,
            "vector_dimension": embedding_result.vector_dimension,
            "cache_fingerprint": embedding_result.cache_fingerprint,
            "candidates_scored": embedding_result.candidates_scored,
            "failures": embedding_result.failures,
            "runtime_seconds": embedding_result.runtime_seconds,
            "cache_hits": embedding_result.cache_hits,
            "cache_misses": embedding_result.cache_misses,
            "error": embedding_result.error,
            "used_for_production_decision": False,
        },
        "embedding_retrieval": {
            "status": (
                embedding_retrieval_result.status
                if embedding_retrieval_result is not None
                else "NOT_APPLICABLE"
            ),
            "requested": bool(
                embedding_retrieval_result
                and embedding_retrieval_result.requested
            ),
            "available": bool(
                embedding_retrieval_result
                and embedding_retrieval_result.available
            ),
            "used": bool(
                embedding_retrieval_result
                and embedding_retrieval_result.used
            ),
            "offers_retrieved": (
                embedding_retrieval_result.offers_retrieved
                if embedding_retrieval_result is not None
                else 0
            ),
            "master_vectors": (
                embedding_retrieval_result.master_vectors
                if embedding_retrieval_result is not None
                else 0
            ),
            "runtime_seconds": (
                embedding_retrieval_result.runtime_seconds
                if embedding_retrieval_result is not None
                else 0.0
            ),
            "cache_hits": (
                embedding_retrieval_result.cache_hits
                if embedding_retrieval_result is not None
                else 0
            ),
            "cache_misses": (
                embedding_retrieval_result.cache_misses
                if embedding_retrieval_result is not None
                else 0
            ),
            "error": (
                embedding_retrieval_result.error
                if embedding_retrieval_result is not None
                else None
            ),
        },
        "agreement_routing": {
            "offers": len(agreement_result.frame),
            "status_counts": {
                str(key): int(value)
                for key, value in agreement_result.frame[
                    "agreement_status"
                ].value_counts().items()
            },
            "route_counts": {
                str(key): int(value)
                for key, value in agreement_result.frame[
                    "routing_decision"
                ].value_counts().items()
            },
            "llm_called": bool(llm_review_result.provider_calls),
            "learning_dataset_modified": False,
            "routes_are_observational": True,
        },
        "llm_review": {
            "enabled": config.llm_review.enabled,
            "status": llm_review_result.status,
            "provider": config.llm_review.provider,
            "model": config.llm_review.model,
            "prompt_version": (
                llm_review_result.results[0].prompt_version
                if llm_review_result.results
                else None
            ),
            "response_schema_version": (
                llm_review_result.results[0].response_schema_version
                if llm_review_result.results
                else None
            ),
            "offers_routed": llm_review_result.offers_routed,
            "provider_calls": llm_review_result.provider_calls,
            "cache_hits": llm_review_result.cache_hits,
            "failures": llm_review_result.failures,
            "error": llm_review_result.error,
            "production_decisions_modified": False,
            "product_master_modified": False,
            "training_data_modified": False,
        },
        "stage_runtimes_seconds": {
            **stage_runtimes,
            "total_before_manifest": time.perf_counter() - run_started,
        },
        "production_input_fingerprint_before": production_fingerprint,
        "production_input_fingerprint_after": _frame_fingerprint(
            production_rows
        ),
        "product_master_fingerprint_before": master_fingerprint,
        "product_master_fingerprint_after": _frame_fingerprint(
            product_master
        ),
        "production_state_unchanged": True,
        "production_outputs_written_by_shadow": [],
        "competitor_discovery_inputs_modified": False,
        "human_reviews_added_to_training": False,
        "challenge_set_opened": False,
        "artifacts": {
            key: {
                "path": str(path),
                "sha256": _sha256_file(path),
            }
            for key, path in output_paths.items()
            if key != "run_manifest"
        },
    }
    _atomic_json(manifest, output_paths["run_manifest"])
    return ShadowRunResult(
        status="COMPLETED_OBSERVATIONAL_ONLY",
        shadow_run_id=effective_run_id,
        prediction_rows=int(len(candidate_frame)),
        offer_groups=int(candidate_frame["offer_group_id"].nunique()),
        failed_shadow_predictions=feature_failures,
        output_paths=output_paths,
    )


def run_shadow_observation_non_blocking(
    production_rows: pd.DataFrame,
    product_master: pd.DataFrame,
    *,
    config: PipelineConfig,
    model_directory: str | Path | None = None,
    shadow_run_id: str | None = None,
    progress: InferenceProgressCallback | None = None,
) -> ShadowRunResult:
    """Fail closed and return production control regardless of shadow errors."""
    if not config.shadow_mode.enabled:
        return run_shadow_observation(
            production_rows,
            product_master,
            config=config,
            model_directory=model_directory,
            shadow_run_id=shadow_run_id,
            progress=progress,
        )
    before_production = production_rows.copy(deep=True)
    before_master = product_master.copy(deep=True)
    try:
        return run_shadow_observation(
            production_rows,
            product_master,
            config=config,
            model_directory=model_directory,
            shadow_run_id=shadow_run_id,
            progress=progress,
        )
    except Exception as error:
        LOGGER.exception(
            "Shadow observation failed closed; production results are retained"
        )
        pd.testing.assert_frame_equal(
            production_rows, before_production, check_exact=True
        )
        pd.testing.assert_frame_equal(
            product_master, before_master, check_exact=True
        )
        failure_directory = config.shadow_mode.output_directory / "failures"
        timestamp = datetime.now(timezone.utc)
        original_exception = capture_exception_details(error)
        failure_path = failure_directory / (
            f"shadow_failure_{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        try:
            _atomic_json(
                {
                    "status": "FAILED_CLOSED",
                    "timestamp": timestamp.isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "original_exception": original_exception,
                    "production_state_unchanged": True,
                    "production_pipeline_must_continue": True,
                    "automatic_production_matching_approved": False,
                },
                failure_path,
            )
            paths = {"failure_manifest": failure_path}
        except Exception:
            LOGGER.exception("Unable to persist shadow failure manifest")
            paths = {}
        return ShadowRunResult(
            status="FAILED_CLOSED",
            shadow_run_id=shadow_run_id,
            prediction_rows=0,
            offer_groups=0,
            failed_shadow_predictions=0,
            output_paths=paths,
            error=str(error),
            original_exception=original_exception,
            exception=error,
        )


def run_configured_shadow_sidecar(
    production_rows: pd.DataFrame,
    product_master: pd.DataFrame,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> ShadowRunResult:
    """Production integration point; disabled in the repository default."""
    config = load_config(config_path)
    if config.ml.mode.value == "shadow":
        config = enable_shadow_for_explicit_model(
            config,
            model_id=str(config.ml.model_id),
        )
    return run_shadow_observation_non_blocking(
        production_rows,
        product_master,
        config=config,
    )


def enable_shadow_for_explicit_model(
    config: PipelineConfig,
    *,
    model_id: str,
    output_directory: Path | None = None,
) -> PipelineConfig:
    """Return an explicit in-memory dry-run config without editing defaults."""
    shadow = replace(
        config.shadow_mode,
        enabled=True,
        model_id=model_id,
        package_reference=None,
        output_directory=output_directory
        if output_directory is not None
        else config.shadow_mode.output_directory,
    )
    return replace(config, shadow_mode=shadow)
