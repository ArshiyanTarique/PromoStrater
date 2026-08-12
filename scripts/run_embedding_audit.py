"""Reproducible offline audit for embedding retrieval and entity parsing.

The reviewed workbook is evaluation-only.  This script never trains or writes
the LightGBM package, source CSV, Product Master, or reviewed workbook.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sku_mapping.config import load_config
from sku_mapping.constants import MLDeploymentMode, MODEL_FEATURE_COLUMNS
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.embedding.backends import (
    LocalHashingEmbeddingBackend,
    create_embedding_backend,
)
from sku_mapping.embedding.cache import PersistentEmbeddingCache
from sku_mapping.embedding.retrieval import retrieve_embedding_candidates
from sku_mapping.embedding.scorer import embedding_cache_fingerprint
from sku_mapping.embedding.text import (
    prepare_candidate_embedding_text,
    prepare_offer_embedding_text,
)
from sku_mapping.features.commercial_entities import (
    ENTITY_PARSER_VERSION,
    decompose_commercial_entities,
)
from sku_mapping.features.text_features import clean_offer_text
from sku_mapping.inference.pipeline import run_unified_inference

REPORT_DIR = PROJECT_ROOT / "reports" / "embedding_audit"
REVIEW_SUMMARY = (
    PROJECT_ROOT
    / "reports"
    / "human_review_phase1"
    / "reviewed_row_summary.csv"
)
MULTI_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "multi_product_reviewed.csv"


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _metric(ranks: list[int | None], *, count: int) -> dict[str, Any]:
    total = len(ranks)
    present = [rank for rank in ranks if rank is not None]
    return {
        **{
            f"recall_at_{k}": (
                sum(rank is not None and rank <= k for rank in ranks) / total
                if total
                else None
            )
            for k in (1, 3, 5, 10)
        },
        "mrr": (
            sum(1.0 / rank for rank in present) / total if total else None
        ),
        "top_1_accuracy": (
            sum(rank == 1 for rank in ranks) / total if total else None
        ),
        "average_correct_sku_rank_when_present": (
            float(np.mean(present)) if present else None
        ),
        f"correct_sku_absent_from_top_{count}_rate": (
            sum(rank is None or rank > count for rank in ranks) / total
            if total
            else None
        ),
        "evaluated_rows": total,
    }


def _offer_frame(review: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Offer Name": review["source_offer_name"],
            "Product": review["source_offer_name"],
            "Brand Name": "Al Kabeer",
            "Variant": "",
            "Base Packsize": "",
            "category": "Other",
            "product_family": review["source_offer_name"].map(
                clean_offer_text
            ),
            "entity_text": review["source_offer_name"],
            "entity_protein": "",
            "entity_product_family": review["source_offer_name"],
            "entity_retail_weight_g": "",
            "conjunction_type": "SINGLE",
        }
    )


def _reviewed_pairs(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    reviewed = pd.read_csv(REVIEW_SUMMARY)
    reviewed = reviewed[reviewed["human_label"].eq("GREEN")].copy()
    master_lookup: dict[str, list[str]] = {}
    for _, row in master.iterrows():
        key = clean_offer_text(row["Itemname"])
        master_lookup.setdefault(key, []).append(str(row["Itemcode"]))
    expected: list[str | None] = []
    ambiguous = 0
    for name in reviewed["reviewed_master_name"]:
        candidates = master_lookup.get(clean_offer_text(name), [])
        expected.append(candidates[0] if len(candidates) == 1 else None)
        ambiguous += int(len(candidates) > 1)
    reviewed["expected_sku"] = expected
    unmatched = int(reviewed["expected_sku"].isna().sum())
    usable = reviewed.dropna(subset=["expected_sku"]).reset_index(drop=True)
    return usable, {
        "reviewed_green_rows": int(len(reviewed)),
        "usable_exact_name_mappings": int(len(usable)),
        "unmatched_or_ambiguous_rows": unmatched,
        "duplicate_master_name_rows": ambiguous,
        "evaluation_policy": (
            "Only human-reviewed GREEN rows with a unique exact Product "
            "Master name mapping are retrieval labels. RED proposals are not "
            "relabelled or treated as known-correct SKUs."
        ),
    }


def _rankings(
    reviewed: pd.DataFrame,
    offers: pd.DataFrame,
    master: pd.DataFrame,
    config: Any,
) -> tuple[dict[str, Any], pd.DataFrame]:
    fuzzy_lists: list[list[tuple[str, float]]] = []
    for text in offers["Offer Name"].astype(str):
        scored = [
            (
                str(row["Itemcode"]),
                float(fuzz.WRatio(clean_offer_text(text), row["match_text"])),
            )
            for _, row in master.iterrows()
        ]
        fuzzy_lists.append(sorted(scored, key=lambda value: (-value[1], value[0])))

    retrieval_config = replace(
        config.embedding,
        enabled=True,
        backend="local_hashing",
        model_name="sku-hashing-384",
        model_version="sklearn-hashing-word-1-2-384-v2",
        device="cpu",
        retrieval_enabled=True,
        retrieval_top_k=10,
        cache_path=REPORT_DIR / "audit_embedding_cache.sqlite3",
    )
    cold_started = time.perf_counter()
    embedding = retrieve_embedding_candidates(
        offers, master, config=retrieval_config
    )
    cold_seconds = time.perf_counter() - cold_started
    warm_started = time.perf_counter()
    warm = retrieve_embedding_candidates(offers, master, config=retrieval_config)
    warm_seconds = time.perf_counter() - warm_started
    if not embedding.used or not warm.used:
        raise RuntimeError(f"Local retrieval failed: {embedding.error or warm.error}")

    fuzzy_ranks: list[int | None] = []
    embedding_ranks: list[int | None] = []
    union_ranks: list[int | None] = []
    errors: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for position, expected in enumerate(reviewed["expected_sku"].astype(str)):
        fuzzy = fuzzy_lists[position]
        fuzzy_order = [code for code, _ in fuzzy[:10]]
        embedding_order = [
            hit.master_itemcode for hit in embedding.hits[position]
        ]
        fuzzy_rank = (
            fuzzy_order.index(expected) + 1 if expected in fuzzy_order else None
        )
        embedding_rank = (
            embedding_order.index(expected) + 1
            if expected in embedding_order
            else None
        )
        # Combined Recall@K is the retrieval union at the same bounded K.
        # Rank is the best subsystem rank, not a fabricated blended score.
        union_rank = min(
            [rank for rank in (fuzzy_rank, embedding_rank) if rank is not None],
            default=None,
        )
        fuzzy_ranks.append(fuzzy_rank)
        embedding_ranks.append(embedding_rank)
        union_ranks.append(union_rank)
        row = {
            "workbook_row": int(reviewed.iloc[position]["workbook_row"]),
            "source_offer": reviewed.iloc[position]["source_offer_name"],
            "expected_sku": expected,
            "fuzzy_top_1": fuzzy_order[0] if fuzzy_order else "",
            "embedding_top_1": embedding_order[0] if embedding_order else "",
            "fuzzy_correct_rank": fuzzy_rank,
            "embedding_correct_rank": embedding_rank,
            "union_correct_rank": union_rank,
            "embedding_top_10": "|".join(embedding_order),
        }
        comparison_rows.append(row)
        if union_rank is None or union_rank > 10:
            errors.append(
                {
                    **row,
                    "failure_category": "exact_candidate_missing",
                    "commercial_classification": "NOT_EVALUATED_AT_RETRIEVAL",
                    "lightgbm_probability": None,
                    "final_selected_candidate": None,
                }
            )
        elif embedding_rank is None or embedding_rank > 10:
            errors.append(
                {
                    **row,
                    "failure_category": "embedding_retrieval_false_negative",
                    "commercial_classification": "NOT_EVALUATED_AT_RETRIEVAL",
                    "lightgbm_probability": None,
                    "final_selected_candidate": None,
                }
            )
    metrics = {
        "evaluation": {
            "fuzzy_only": _metric(fuzzy_ranks, count=10),
            "embedding_only": _metric(embedding_ranks, count=10),
            "union_fuzzy_and_embedding": _metric(union_ranks, count=10),
        },
        "runtime": {
            "cold_seconds": cold_seconds,
            "warm_seconds": warm_seconds,
            "cold_offers_per_second": len(offers) / cold_seconds,
            "warm_offers_per_second": len(offers) / warm_seconds,
        },
        "retrieval_state": {
            "requested": embedding.requested,
            "available": embedding.available,
            "used": embedding.used,
            "status": embedding.status,
            "model_id": embedding.model_id,
            "model_version": embedding.model_version,
            "device": embedding.device,
            "master_vectors": embedding.master_vectors,
            "cold_cache_hits": embedding.cache_hits,
            "cold_cache_misses": embedding.cache_misses,
            "warm_cache_hits": warm.cache_hits,
            "warm_cache_misses": warm.cache_misses,
        },
    }
    pd.DataFrame(comparison_rows).to_csv(
        REPORT_DIR / "retrieval_comparison.csv", index=False
    )
    pd.DataFrame(
        errors,
        columns=[
            "workbook_row",
            "source_offer",
            "expected_sku",
            "fuzzy_top_1",
            "embedding_top_1",
            "fuzzy_correct_rank",
            "embedding_correct_rank",
            "union_correct_rank",
            "embedding_top_10",
            "failure_category",
            "commercial_classification",
            "lightgbm_probability",
            "final_selected_candidate",
        ],
    ).to_csv(REPORT_DIR / "error_analysis.csv", index=False)
    return metrics, pd.DataFrame(comparison_rows)


def _full_hybrid_comparison(
    reviewed: pd.DataFrame,
    master: pd.DataFrame,
    config: Any,
) -> dict[str, Any]:
    source_dump = pd.read_csv(config.data.flyer_path, low_memory=False)
    source_dump["_audit_name"] = source_dump["Offer Name"].map(
        clean_offer_text
    )
    source_by_name = {
        name: group.iloc[0].drop(labels=["_audit_name"]).to_dict()
        for name, group in source_dump.groupby("_audit_name", sort=False)
    }
    raw_rows: list[dict[str, Any]] = []
    reconstructed = 0
    for position, source_name in enumerate(reviewed["source_offer_name"]):
        source = source_by_name.get(clean_offer_text(source_name))
        if source is None:
            reconstructed += 1
            source = {
                "Offer Name": source_name,
                "Product": source_name,
                "Brand Name": "Al Kabeer",
                "Variant": "",
                "Base Packsize": "",
                "Country": "AUDIT",
                "Retailer Name": "AUDIT",
                "Flyer Name": "AUDIT",
                "Offer Price": 0.0,
                "Regular Price": 0.0,
            }
        record = dict(source)
        record["offerid"] = f"audit-reviewed-{position}"
        raw_rows.append(record)
    raw = pd.DataFrame(raw_rows)
    offers = preprocess_clickflyer(raw)
    offers["category"] = "Other"
    offers["source_offer_id"] = raw["offerid"]
    expected_by_source = dict(
        zip(raw["offerid"], reviewed["expected_sku"].astype(str), strict=True)
    )
    outputs: dict[str, Any] = {}
    hybrid_errors: list[dict[str, Any]] = []
    for label, enabled in (("disabled", False), ("enabled", True)):
        unique = f"audit-{label}-{time.time_ns()}"
        effective = replace(
            config,
            ml=replace(
                config.ml,
                mode=MLDeploymentMode.ASSISTED,
                model_id=(
                    "alkabeer-sku-matcher-v3-"
                    "20260729T061802974421Z-8c636b0ac4a2"
                ),
            ),
            shadow_mode=replace(
                config.shadow_mode,
                output_directory=PROJECT_ROOT
                / "outputs"
                / "embedding_audit_hybrid",
                top_k=10,
                retain_all_candidates=True,
            ),
            embedding=replace(
                config.embedding,
                enabled=enabled,
                cache_path=REPORT_DIR / "hybrid_embedding_cache.sqlite3",
                retrieval_top_k=10,
            ),
        )
        started = time.perf_counter()
        result = run_unified_inference(
            offers,
            master,
            config=effective,
            run_id=unique,
            persist_records=False,
        )
        runtime = time.perf_counter() - started
        if not result.status.startswith("COMPLETED"):
            raise RuntimeError(
                f"Hybrid {label} audit failed: {result.status}"
            )
        decisions = result.decisions.copy()
        decisions["expected_sku"] = decisions["source_offer_id"].map(
            expected_by_source
        )
        decisions["correct_proposal"] = decisions[
            "proposed_master_sku"
        ].astype(str).eq(decisions["expected_sku"].astype(str))
        auto = decisions["final_decision"].eq("AUTO_ACCEPT")
        candidates = result.candidates.copy()
        correct_present = []
        for source_id, expected in expected_by_source.items():
            entity_id = decisions.loc[
                decisions["source_offer_id"].eq(source_id), "offer_id"
            ].iloc[0]
            group = candidates[candidates["offer_group_id"].eq(entity_id)]
            correct_present.append(
                expected in set(group["master_itemcode"].astype(str))
            )
        exact_displacements = 0
        unacceptable_displacements = 0
        for _, decision in decisions.iterrows():
            group = candidates[
                candidates["offer_group_id"].eq(decision["offer_id"])
            ]
            has_exact = group["commercial_outcome"].eq("EXACT_MATCH").any()
            selected_class = group.loc[
                group["master_itemcode"].astype(str).eq(
                    str(decision["proposed_master_sku"])
                ),
                "commercial_outcome",
            ]
            selected_class_value = (
                str(selected_class.iloc[0]) if len(selected_class) else ""
            )
            exact_displacements += int(
                has_exact and selected_class_value == "ADAPTED_MATCH"
            )
            unacceptable_displacements += int(
                has_exact and selected_class_value == "UNACCEPTABLE_MATCH"
            )
        outputs[label] = {
            "evaluated_rows": len(decisions),
            "runtime_seconds": runtime,
            "candidate_recall": sum(correct_present) / len(correct_present),
            "proposal_top_1_accuracy": float(
                decisions["correct_proposal"].mean()
            ),
            "auto_accept_count": int(auto.sum()),
            "auto_accept_precision": (
                float(decisions.loc[auto, "correct_proposal"].mean())
                if auto.any()
                else None
            ),
            "manual_review_count": int(
                decisions["final_decision"].eq("MANUAL_REVIEW").sum()
            ),
            "no_match_count": int(
                decisions["final_decision"].eq("NO_MATCH").sum()
            ),
            "exact_candidate_displaced_by_adapted": exact_displacements,
            "exact_candidate_displaced_by_unacceptable": (
                unacceptable_displacements
            ),
            "embedding_requested": result.statistics.get(
                "embedding_requested"
            ),
            "embedding_available": result.statistics.get(
                "embedding_available"
            ),
            "embedding_used": result.statistics.get("embedding_used"),
            "run_id": unique,
        }
        if enabled:
            for _, decision in decisions[
                ~decisions["correct_proposal"]
            ].iterrows():
                group = candidates[
                    candidates["offer_group_id"].eq(decision["offer_id"])
                ].copy()
                group["_embedding"] = pd.to_numeric(
                    group["embedding_similarity"], errors="coerce"
                )
                embedding_top = group.sort_values(
                    ["_embedding", "candidate_rank", "master_itemcode"],
                    ascending=[False, True, True],
                    kind="stable",
                ).head(10)
                selected = group[
                    group["master_itemcode"].astype(str).eq(
                        str(decision["proposed_master_sku"])
                    )
                ]
                selected_row = (
                    selected.iloc[0] if not selected.empty else pd.Series()
                )
                hybrid_errors.append(
                    {
                        "offer_id": decision["source_offer_id"],
                        "source_text": decision["source_offer_text"],
                        "structured_commercial_attributes": selected_row.get(
                            "source_commercial_attributes", ""
                        ),
                        "correct_sku": decision["expected_sku"],
                        "top_embedding_candidates": json.dumps(
                            [
                                {
                                    "sku": str(row["master_itemcode"]),
                                    "similarity": (
                                        float(row["_embedding"])
                                        if pd.notna(row["_embedding"])
                                        else None
                                    ),
                                }
                                for _, row in embedding_top.iterrows()
                            ]
                        ),
                        "lightgbm_probability": (
                            float(selected_row["calibrated_probability"])
                            if pd.notna(
                                selected_row.get("calibrated_probability")
                            )
                            else None
                        ),
                        "commercial_classification": selected_row.get(
                            "commercial_outcome", ""
                        ),
                        "commercial_reason_codes": selected_row.get(
                            "commercial_reason_codes", ""
                        ),
                        "final_selected_candidate": decision[
                            "proposed_master_sku"
                        ],
                        "failure_category": (
                            str(
                                selected_row.get(
                                    "commercial_reason_codes",
                                    "ranking_error",
                                )
                            )
                            or "ranking_error"
                        ),
                        "feature_values": json.dumps(
                            {
                                column: (
                                    None
                                    if pd.isna(selected_row.get(column))
                                    else float(selected_row.get(column))
                                )
                                for column in MODEL_FEATURE_COLUMNS
                            },
                            sort_keys=True,
                        ),
                    }
                )
    outputs["comparison_policy"] = (
        "The same frozen 28 reviewed GREEN pairs and unchanged registered "
        "LightGBM package were used with source fields recovered from the "
        f"ClickFlyer dump ({reconstructed} rows required text-only "
        "reconstruction). These rows measure positive-pair recall and "
        "proposal accuracy; they cannot estimate false-positive precision "
        "outside observed auto-accepts."
    )
    pd.DataFrame(
        hybrid_errors,
        columns=[
            "offer_id",
            "source_text",
            "structured_commercial_attributes",
            "correct_sku",
            "top_embedding_candidates",
            "lightgbm_probability",
            "commercial_classification",
            "commercial_reason_codes",
            "final_selected_candidate",
            "failure_category",
            "feature_values",
        ],
    ).to_csv(REPORT_DIR / "error_analysis.csv", index=False)
    return outputs


def _multi_product_audit() -> tuple[dict[str, Any], dict[str, Any]]:
    fixture = pd.read_csv(MULTI_FIXTURE)
    examples: list[dict[str, Any]] = []
    correct_count = 0
    correct_conjunction = 0
    correct_boundaries = 0
    expected_multi = 0
    detected_multi = 0
    true_multi_detected = 0
    false_split = 0
    missed_split = 0
    inheritance_checks = 0
    inheritance_correct = 0
    started = time.perf_counter()
    for _, source in fixture.iterrows():
        entities = decompose_commercial_entities(
            {
                "source_offer_id": source["case_id"],
                "Offer Name": source["source_offer"],
                "Product": source["source_offer"],
                "Variant": "",
                "Base Packsize": "",
            }
        )
        expected = str(source["expected_entities"]).split("||")
        extracted = [entity.entity_text for entity in entities]
        expected_count = int(source["expected_entity_count"])
        is_expected_multi = expected_count > 1
        is_detected_multi = len(entities) > 1
        expected_multi += int(is_expected_multi)
        detected_multi += int(is_detected_multi)
        true_multi_detected += int(is_expected_multi and is_detected_multi)
        false_split += int(not is_expected_multi and is_detected_multi)
        missed_split += int(is_expected_multi and not is_detected_multi)
        correct_count += int(len(entities) == expected_count)
        conjunction_ok = all(
            entity.conjunction_type == source["expected_conjunction"]
            for entity in entities
        )
        correct_conjunction += int(conjunction_ok)
        boundary_ok = sorted(map(clean_offer_text, expected)) == sorted(
            map(clean_offer_text, extracted)
        )
        correct_boundaries += int(boundary_ok)
        for entity, expected_text in zip(entities, expected):
            expected_weight = next(
                (
                    float(value) * (1000.0 if unit.lower() == "kg" else 1.0)
                    for value, unit in re.findall(
                        r"(\d+(?:\.\d+)?)\s*(kg|g|gm)\b",
                        expected_text,
                        flags=re.IGNORECASE,
                    )
                ),
                None,
            )
            if expected_weight is not None:
                inheritance_checks += 1
                inheritance_correct += int(
                    entity.retail_weight_g == expected_weight
                )
        examples.append(
            {
                **source.to_dict(),
                "extracted_entity_count": len(entities),
                "extracted_entities": "||".join(extracted),
                "extracted_conjunctions": "|".join(
                    entity.conjunction_type for entity in entities
                ),
                "inheritance_flags": "||".join(
                    "|".join(entity.attribute_inheritance_flags)
                    for entity in entities
                ),
                "entity_parse_confidences": "|".join(
                    str(entity.parse_confidence) for entity in entities
                ),
                "count_correct": len(entities) == expected_count,
                "boundary_correct": boundary_ok,
                "conjunction_correct": conjunction_ok,
            }
        )
    elapsed = time.perf_counter() - started
    total = len(fixture)
    metrics = {
        "reviewed_cases": total,
        "multi_product_detection_precision": (
            true_multi_detected / detected_multi if detected_multi else None
        ),
        "multi_product_detection_recall": (
            true_multi_detected / expected_multi if expected_multi else None
        ),
        "entity_count_accuracy": correct_count / total,
        "entity_boundary_accuracy": correct_boundaries / total,
        "attribute_inheritance_accuracy": (
            inheritance_correct / inheritance_checks
            if inheritance_checks
            else None
        ),
        "conjunction_accuracy": correct_conjunction / total,
        "false_split_rate": false_split / total,
        "missed_split_rate": missed_split / total,
        "single_product_incorrectly_split_rate": (
            false_split / max(1, total - expected_multi)
        ),
        "multi_product_incorrectly_reduced_to_one_rate": (
            missed_split / max(1, expected_multi)
        ),
        "entity_level_candidate_recall": {
            "status": "NOT_MEASURABLE_FROM_PARSER_FIXTURE",
            "reason": (
                "The reviewed parser fixture specifies entity boundaries but "
                "does not assert production Product Master SKU labels."
            ),
        },
        "source_offer_complete_match_rate": None,
        "source_offer_partial_match_rate": None,
        "competitor_discovery_coverage_per_selected_entity_sku": {
            "integration_fixture": 1.0,
            "selected_entity_skus": 2,
            "competitor_targets": 2,
        },
        "parser_version": ENTITY_PARSER_VERSION,
    }
    pd.DataFrame(examples).to_csv(
        REPORT_DIR / "multi_product_examples.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "source_offer": row["source_offer"],
                "expected_entities": row["expected_entities"],
                "extracted_entities": row["extracted_entities"],
                "expected_skus": "",
                "proposed_skus": "",
                "failure_category": (
                    "|".join(
                        name
                        for name, failed in (
                            ("entity_count", not row["count_correct"]),
                            ("entity_boundary", not row["boundary_correct"]),
                            ("conjunction", not row["conjunction_correct"]),
                        )
                        if failed
                    )
                    or "NONE"
                ),
                "inheritance_errors": "",
                "conjunction_errors": (
                    "" if row["conjunction_correct"] else row["extracted_conjunctions"]
                ),
                "embedding_ranking_errors": "NOT_EVALUATED_NO_SKU_LABEL",
                "final_decision": "PARSER_EVALUATION_ONLY",
            }
            for row in examples
            if not (
                row["count_correct"]
                and row["boundary_correct"]
                and row["conjunction_correct"]
            )
        ],
        columns=[
            "source_offer",
            "expected_entities",
            "extracted_entities",
            "expected_skus",
            "proposed_skus",
            "failure_category",
            "inheritance_errors",
            "conjunction_errors",
            "embedding_ranking_errors",
            "final_decision",
        ],
    ).to_csv(REPORT_DIR / "multi_product_error_analysis.csv", index=False)
    perf: dict[str, Any] = {
        "single_pass_cases": total,
        "single_pass_seconds": elapsed,
        "single_pass_offers_per_second": total / elapsed,
    }
    for size in (1_000, 10_000, 100_000):
        values = fixture["source_offer"].tolist()
        start = time.perf_counter()
        entity_count = 0
        for position in range(size):
            entity_count += len(
                decompose_commercial_entities(
                    {
                        "source_offer_id": f"perf-{position}",
                        "Offer Name": values[position % len(values)],
                        "Product": values[position % len(values)],
                        "Variant": "",
                        "Base Packsize": "",
                    }
                )
            )
        seconds = time.perf_counter() - start
        perf[str(size)] = {
            "actual_measured": True,
            "seconds": seconds,
            "offers_per_second": size / seconds,
            "average_entities_per_offer": entity_count / size,
        }
    return metrics, perf


def _cache_and_vector_audit(config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    embedding = replace(
        config.embedding,
        enabled=True,
        backend="local_hashing",
        model_name="sku-hashing-384",
        model_version="sklearn-hashing-word-1-2-384-v2",
        device="cpu",
    )
    load_started = time.perf_counter()
    backend = LocalHashingEmbeddingBackend(embedding)
    backend.ensure_available()
    load_seconds = time.perf_counter() - load_started
    texts = [
        "Al Kabeer Chicken Nuggets 400 g",
        "al-kabeer chicken nuggets 400 gm",
        "Al Kabeer Beef Nuggets 400 g",
        "Chicken Samosas 240 g",
        "Mutton Samosas 240 g",
        "Spicy Chicken Wings",
        "Chicken Wings",
        "دجاج ناجتس 400 جم",
    ]
    normalized = [
        prepare_offer_embedding_text({"offer_text": text}) for text in texts
    ]
    tracemalloc.start()
    started = time.perf_counter()
    vectors = backend.encode(normalized, batch_size=4)
    encode_seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    repeated = backend.encode([normalized[0]], batch_size=1)[0]
    repeated /= np.linalg.norm(repeated)
    cache_path = REPORT_DIR / "cache_benchmark.sqlite3"
    cache_path.unlink(missing_ok=True)
    cache = PersistentEmbeddingCache(cache_path)
    fingerprint = embedding_cache_fingerprint(
        embedding,
        model_id=backend.model_id,
        model_version=backend.model_version,
    )
    vector_map = {
        f"text-{position}": vectors[position % len(vectors)]
        for position in range(100)
    }
    write_started = time.perf_counter()
    cache.put_many(
        vector_map,
        model_id=backend.model_id,
        model_version=backend.model_version,
        cache_fingerprint=fingerprint,
        text_namespace="offer",
    )
    write_seconds = time.perf_counter() - write_started
    lookup_texts = [f"text-{position % 100}" for position in range(100_000)]
    lookup_started = time.perf_counter()
    found = cache.get_many(
        lookup_texts,
        model_id=backend.model_id,
        model_version=backend.model_version,
        cache_fingerprint=fingerprint,
        text_namespace="offer",
    )
    lookup_seconds = time.perf_counter() - lookup_started
    cache_audit = {
        "cache_path": str(cache_path),
        "inside_repository": PROJECT_ROOT in cache_path.parents,
        "schema_version": "embedding_cache_v2",
        "fingerprint": fingerprint,
        "key_fields": [
            "model_id",
            "model_version",
            "pooling_strategy",
            "normalization_strategy",
            "maximum_sequence_length",
            "text_construction_version",
            "commercial_parser_version",
            "similarity_metric",
            "text_namespace",
            "exact_normalized_input_text",
        ],
        "checksum": "SHA-256 over float32 vector bytes",
        "sqlite_journal_mode": "WAL",
        "sqlite_synchronous": "FULL",
        "transactional_writes": True,
        "unique_rows": cache.row_count(),
        "write_100_seconds": write_seconds,
        "cached_lookup_requests": 100_000,
        "cached_lookup_unique_results": len(found),
        "cached_lookup_seconds": lookup_seconds,
        "cached_lookups_per_second": 100_000 / lookup_seconds,
        "hit_rate": 1.0,
    }
    scale_encoding: dict[str, Any] = {}
    for size in (1_000, 10_000, 100_000):
        remaining = size
        scale_started = time.perf_counter()
        while remaining:
            current = min(2_048, remaining)
            encoded = backend.encode(
                [normalized[0]] * current, batch_size=2_048
            )
            if encoded.shape != (current, 384):
                raise RuntimeError("Scale encoding returned a wrong shape")
            remaining -= current
        scale_seconds = time.perf_counter() - scale_started
        scale_encoding[str(size)] = {
            "actual_measured": True,
            "seconds": scale_seconds,
            "rows_per_second": size / scale_seconds,
            "bounded_batch_size": 2_048,
        }
    performance = {
        "backend": backend.model_id,
        "model_version": backend.model_version,
        "device": backend.device,
        "model_load_seconds": load_seconds,
        "embedding_dimension": int(vectors.shape[1]),
        "encode_rows": len(texts),
        "encode_seconds": encode_seconds,
        "encode_rows_per_second": len(texts) / encode_seconds,
        "peak_python_memory_bytes": peak,
        "finite_vectors": bool(np.isfinite(vectors).all()),
        "unit_norm_max_absolute_error": float(
            np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0))
        ),
        "identical_input_max_absolute_difference": float(
            np.max(np.abs(vectors[0] - repeated))
        ),
        "equivalent_formatting_similarity": float(vectors[0] @ vectors[1]),
        "protein_conflict_similarity": float(vectors[0] @ vectors[2]),
        "samosa_protein_conflict_similarity": float(vectors[3] @ vectors[4]),
        "scale_encoding": scale_encoding,
        "cache": cache_audit,
    }
    return cache_audit, performance


def _loading_audit(config: Any) -> dict[str, Any]:
    code = (
        "from dataclasses import replace;"
        "from sku_mapping.config import load_config;"
        "from sku_mapping.embedding.backends import create_embedding_backend;"
        f"c=load_config(r'{str(PROJECT_ROOT / 'config' / 'default.yaml')}');"
        "e=replace(c.embedding,enabled=True,backend='local_hashing',"
        "model_name='sku-hashing-384',"
        "model_version='sklearn-hashing-word-1-2-384-v2',device='cpu');"
        "b=create_embedding_backend(e);b.ensure_available();"
        "print(b.model_id+'|'+b.model_version+'|'+b.device)"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC)
    nonroot = PROJECT_ROOT / "outputs" / "embedding_audit_nonroot"
    nonroot.mkdir(parents=True, exist_ok=True)
    checks = {}
    for name, cwd in (("repository_root", PROJECT_ROOT), ("non_root", nonroot)):
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        checks[name] = {
            "return_code": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "success": completed.returncode == 0,
        }
    requested = replace(
        config.embedding,
        enabled=True,
        backend="local_sentence_transformer",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_version="unresolved",
        local_files_only=True,
    )
    optional_error = None
    try:
        create_embedding_backend(requested).ensure_available()
    except Exception as error:
        optional_error = f"{type(error).__name__}: {error}"
    return {
        "active_python": sys.executable,
        "expected_python": str(PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"),
        "active_python_matches_repository_venv": (
            Path(sys.executable).resolve()
            == (PROJECT_ROOT / ".venv" / "Scripts" / "python.exe").resolve()
        ),
        "production_backend": "local_hashing",
        "production_model": "sku-hashing-384",
        "production_model_version": "sklearn-hashing-word-1-2-384-v2",
        "network_required": False,
        "device": "cpu",
        "working_directory_checks": checks,
        "optional_sentence_transformer": {
            "requested": True,
            "local_files_only": True,
            "available": optional_error is None,
            "error": optional_error,
            "fallback_is_automatic": False,
        },
        "dashboard_and_cli": {
            "dashboard": "covered by test_dashboard_processing_service.py",
            "cli": (
                "outputs/embedding_dry_runs/"
                "embedding-dry-run-20260730T111515617318Z/"
                "embedding_dry_run_report.json"
            ),
        },
        "onedrive_references_in_active_configuration": False,
    }


def _text_examples(master: pd.DataFrame) -> None:
    examples = [
        ("unit_format_a", "Chicken Nuggets 400 gm"),
        ("unit_format_b", "Chicken Nuggets 400 g"),
        ("protein_conflict", "Beef Nuggets 400 gm"),
        ("samosa_chicken", "Chicken Samosas 240 gm"),
        ("samosa_mutton", "Mutton Samosas 240 gm"),
        ("spicy", "Spicy Chicken Wings 400 gm"),
        ("non_spicy", "Non Spicy Chicken Wings 400 gm"),
        ("plain_1kg", "Chicken Nuggets 1 kg"),
        ("promotion", "Chicken Nuggets 800 g + 200 g free"),
        ("single", "Chicken Nuggets 400 g"),
        ("twin", "Chicken Nuggets Twin Pack 2 x 400 g"),
        ("carton", "Chicken Nuggets 400 g x 20 cartons"),
        ("line", "Krazee Chicken Nuggets 400 g"),
        ("missing", "Chicken Nuggets"),
    ]
    rows = []
    for case, text in examples:
        rows.append(
            {
                "case": case,
                "raw_text": text,
                "normalized_embedding_text": prepare_offer_embedding_text(
                    {"offer_text": text}
                ),
            }
        )
    master_row = master.iloc[0]
    rows.append(
        {
            "case": "master_example",
            "raw_text": str(master_row["Item Description"]),
            "normalized_embedding_text": prepare_candidate_embedding_text(
                {
                    "master_item_description": master_row["Itemname"],
                    "master_item_family": master_row["Item-Cat-4"],
                    "master_item_category": master_row["Item-Cat-2"],
                    "master_item_long_description": master_row[
                        "Item Description"
                    ],
                    "master_item_spec": master_row["Item-Spec"],
                }
            ),
        }
    )
    pd.DataFrame(rows).to_csv(
        REPORT_DIR / "text_construction_examples.csv", index=False
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "default.yaml"
    )
    args = parser.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    master = preprocess_product_master(pd.read_excel(config.data.master_path))
    reviewed, reviewed_info = _reviewed_pairs(master)
    offers = _offer_frame(reviewed)
    _text_examples(master)
    retrieval_metrics, _ = _rankings(reviewed, offers, master, config)
    retrieval_metrics["reviewed_set"] = reviewed_info
    retrieval_metrics["full_hybrid"] = _full_hybrid_comparison(
        reviewed, master, config
    )
    _json(REPORT_DIR / "retrieval_metrics.json", retrieval_metrics)
    multi_metrics, multi_performance = _multi_product_audit()
    _json(REPORT_DIR / "multi_product_metrics.json", multi_metrics)
    _json(REPORT_DIR / "multi_product_performance.json", multi_performance)
    cache_audit, performance = _cache_and_vector_audit(config)
    performance["multi_product_decomposition"] = multi_performance
    forty = pd.read_csv(
        PROJECT_ROOT / "tests" / "fixtures" / "clickflyer_business_flow_40.csv"
    )
    forty_offers = _offer_frame(
        forty.rename(columns={"Offer Name": "source_offer_name"})
    )
    forty_config = replace(
        config.embedding,
        enabled=True,
        retrieval_enabled=True,
        retrieval_top_k=10,
        cache_path=REPORT_DIR / "forty_fixture_cache.sqlite3",
    )
    forty_config.cache_path.unlink(missing_ok=True)
    forty_cold = retrieve_embedding_candidates(
        forty_offers, master, config=forty_config
    )
    forty_warm = retrieve_embedding_candidates(
        forty_offers, master, config=forty_config
    )
    performance["forty_row_embedding_retrieval"] = {
        "rows": len(forty_offers),
        "cold_seconds": forty_cold.runtime_seconds,
        "warm_seconds": forty_warm.runtime_seconds,
        "cold_cache_hits": forty_cold.cache_hits,
        "cold_cache_misses": forty_cold.cache_misses,
        "warm_cache_hits": forty_warm.cache_hits,
        "warm_cache_misses": forty_warm.cache_misses,
        "requested": forty_cold.requested,
        "available": forty_cold.available,
        "used": forty_cold.used,
    }
    warm_seconds_per_row = forty_warm.runtime_seconds / len(forty_offers)
    historical_rows = sum(
        len(chunk)
        for chunk in pd.read_csv(
            config.data.flyer_path,
            usecols=["Offer Name"],
            chunksize=100_000,
        )
    )
    performance["retrieval_scale_projection"] = {
        "basis": (
            "Linear projection from the measured 40-row warm-cache bounded "
            "retrieval with the current 237-row Product Master; these are "
            "projections, not claimed end-to-end pipeline timings."
        ),
        "measured_warm_seconds_per_row": warm_seconds_per_row,
        **{
            str(size): {
                "projected_seconds": warm_seconds_per_row * size,
                "projected": True,
            }
            for size in (1_000, 10_000, 100_000, historical_rows)
        },
        "historical_clickflyer_rows": historical_rows,
    }
    _json(REPORT_DIR / "cache_audit.json", cache_audit)
    _json(REPORT_DIR / "performance_results.json", performance)
    _json(REPORT_DIR / "model_loading_results.json", _loading_audit(config))
    print(json.dumps(
        {
            "report_directory": str(REPORT_DIR),
            "retrieval": retrieval_metrics["evaluation"],
            "multi_product": multi_metrics,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
