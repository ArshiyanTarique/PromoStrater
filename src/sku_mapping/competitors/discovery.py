"""Master-SKU-centric competitor discovery over one ClickFlyer dump.

The internal long-form table records every evaluated relationship in the
uploaded dump.  The user-facing aggregate is derived only from supported rows
in that table and therefore cannot develop misaligned lists.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from sku_mapping.competitors.policy import DECISION_COLUMNS
from sku_mapping.competitors.text_normalisation import (
    COMPETITOR_TOKEN_ALIASES,
    strip_competitor_brand,
)
from sku_mapping.config import CompetitorConfig
from sku_mapping.data.preprocessing import normalize_product_family
from sku_mapping.features.measurement_features import pack_is_compatible
from sku_mapping.features.semantic_features import (
    _family_concept_set,
    _family_set,
    _protein_set,
)

COMPETITOR_EXPORT_COLUMNS = (
    "master_sku",
    "master_name",
    "master_description",
    "source_alkabeer_offer_ids",
    "source_entity_ids",
    "source_alkabeer_offer_names",
    "competitor_count",
    "competitor_brand_names",
    "competitor_offer_ids",
    "competitor_offer_names",
    "competitor_products",
    "competitor_variants",
    "competitor_pack_sizes",
    "competitor_retailers",
    "competitor_flyers",
    "competitor_offer_prices",
    "competitor_regular_prices",
    "competitor_status",
    "competitor_reason",
    "run_id",
)

COMPETITOR_LONG_COLUMNS = (
    "master_sku",
    "master_name",
    "master_description",
    "source_alkabeer_offer_id",
    "source_alkabeer_offer_name",
    "competitor_offer_id",
    "competitor_offer_name",
    "competitor_product",
    "competitor_brand",
    "competitor_variant",
    "competitor_pack_size",
    "competitor_retailer",
    "competitor_flyer",
    "competitor_offer_price",
    "competitor_regular_price",
    "competitor_match_score",
    "competitor_adjusted_score",
    "competitor_match_status",
    "competitor_match_reason",
    "competitor_rank",
    # Diagnostics only. The learned signal is recorded so a reviewer can see
    # what reordered a list, and is deliberately absent from
    # COMPETITOR_EXPORT_COLUMNS - the business export states decisions, not
    # model internals. The score is a raw margin, comparable only within one
    # offer's shortlist; see competitors.reranker.
    "competitor_lightgbm_score",
    "competitor_lightgbm_rank",
    "competitor_ranking_source",
    "run_id",
    # The terminal automatic verdict. Always present so the audit contract does
    # not change shape with configuration: when automatic decisions are off the
    # three columns are empty rather than absent, which kept the export
    # validator and this tuple in agreement.
    "competitor_decision",
    "competitor_decision_reason",
    "competitor_decision_source",
)

SUPPORTED_COMPETITOR_STATUSES = frozenset({"MATCHED", "AMBIGUOUS"})

LOGGER = logging.getLogger(__name__)
COMPETITOR_EVALUATION_CHUNK_SIZE = 10_000
CompetitorProgressCallback = Callable[[int, int, int, int, str], None]


@dataclass(frozen=True)
class CompetitorDiscoveryResult:
    """One target aggregate, evaluated relationship audit, and exact counts."""

    export: pd.DataFrame
    long_format: pd.DataFrame
    long_format_path: Path | None
    eligible_target_count: int
    diagnostics: dict[str, object] = field(default_factory=dict)


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_scalar(value: object) -> object:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_list(values: list[object]) -> str:
    return json.dumps(
        [_json_scalar(value) for value in values],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _require_columns(
    frame: pd.DataFrame,
    columns: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _target_reason(
    reason_counts: Mapping[str, int], source_count: int
) -> str:
    if source_count == 0:
        return "no competitor offers in upload"
    if not reason_counts:
        return "no competitor offers were available for evaluation"
    if "BELOW_COMPETITOR_ELIGIBILITY_POLICY" in reason_counts:
        return "no candidate exceeded the competitor eligibility policy"
    if len(reason_counts) == 1:
        only_reason = str(next(iter(reason_counts)))
        messages = {
            "PROTEIN_CONFLICT": "all competitor candidates had protein conflicts",
            "FAMILY_CONFLICT": "all competitor candidates had family conflicts",
            "FORM_CONFLICT": (
                "all competitor candidates were a different product form"
            ),
            "PACK_CONFLICT": "all competitor candidates had pack conflicts",
            "PRODUCT_FAMILY_CONFLICT": (
                "no compatible competitor offer in upload; "
                "all source offers had family conflicts"
            ),
            "CATEGORY_CONFLICT": (
                "no compatible competitor offer in upload; "
                "all source offers had category conflicts"
            ),
            "BELOW_COMPETITOR_ELIGIBILITY_POLICY": (
                "no candidate exceeded the competitor eligibility policy"
            ),
        }
        if only_reason in messages:
            return messages[only_reason]
    return "no compatible competitor offer in upload"


def _canonical_sources(prepared_offers: pd.DataFrame) -> pd.DataFrame:
    identity_column = (
        "source_offer_id"
        if "source_offer_id" in prepared_offers
        else "offer_group_id"
    )
    canonical = prepared_offers.drop_duplicates(
        identity_column, keep="first"
    ).copy()
    canonical["_business_offer_id"] = canonical[identity_column].map(
        _safe_text
    )
    if canonical["_business_offer_id"].eq("").any():
        raise ValueError("Competitor source contains an empty offer identity")
    return canonical


#: Re-exported so existing call sites and tests keep resolving. The table and
#: the stripping rule now live in :mod:`competitors.text_normalisation`, which
#: the ML re-ranker shares - see that module for why competitors need it.
_COMPETITOR_TOKEN_ALIASES = COMPETITOR_TOKEN_ALIASES


def _canonical_competitors(prepared_offers: pd.DataFrame) -> pd.DataFrame:
    """Competitor rows, one per (offer identity, variant).

    A ClickFlyer offer is repeated once per variant it covers. Deduplicating on
    the offer id alone kept whichever row sorted first - in practice the
    ``No Variant`` row - and discarded the rest. The variant is part of
    ``match_text``, so dropping the ``Chicken`` row removed the only evidence
    tying that offer to a chicken SKU and left it unmatchable.

    Deliberately separate from :func:`_canonical_sources`, which must stay one
    row per offer id because ``source_lookup`` indexes on it.
    """
    identity_column = (
        "source_offer_id"
        if "source_offer_id" in prepared_offers
        else "offer_group_id"
    )
    pool = prepared_offers[
        ~prepared_offers["is_own"].fillna(False).astype(bool)
    ]
    canonical = pool.drop_duplicates(
        [identity_column, "Variant"], keep="first"
    ).copy()
    canonical["_business_offer_id"] = canonical[identity_column].map(
        _safe_text
    )
    if canonical["_business_offer_id"].eq("").any():
        raise ValueError("Competitor source contains an empty offer identity")
    return canonical


def _competitor_match_text(pool: pd.DataFrame) -> pd.Series:
    """Scoring text for a competitor row.

    Two adjustments to ``match_text``, both scoped to competitor discovery:

    * The competitor's own brand is removed. Scoring compares against Al Kabeer
      master text, where a rival brand can never appear, so those tokens are
      guaranteed misses that only dilute the ratio.
    * Transliterations are folded to one spelling (see
      :data:`_COMPETITOR_TOKEN_ALIASES`).

    A brand token that also appears in the row's ``Product`` is kept - it is
    carrying product meaning, not just branding.
    """
    texts = pool["match_text"].map(_safe_text)
    brands = pool["Brand Name"].map(_safe_text)
    products = pool["Product"].map(_safe_text)

    cleaned = [
        strip_competitor_brand(text, brand, protected=product)
        for text, brand, product in zip(texts, brands, products)
    ]
    return pd.Series(cleaned, index=pool.index, dtype="object")


def _master_profiles(product_master: pd.DataFrame) -> pd.DataFrame:
    profiles = product_master.copy()
    profiles["_business_master_sku"] = profiles["Itemcode"].map(_safe_text)
    profiles = profiles[
        profiles["_business_master_sku"].ne("")
    ].drop_duplicates("_business_master_sku", keep="first")
    profiles["_business_family"] = profiles.apply(
        lambda row: normalize_product_family(
            row["Item-Cat-4"] or row["Itemname"]
        ),
        axis=1,
    )
    profiles["_business_target_text"] = profiles.apply(
        lambda row: " ".join(
            _safe_text(row.get(column))
            for column in (
                "Itemname",
                "Item-Cat-4",
                "Item Description",
            )
            if _safe_text(row.get(column))
        ).lower(),
        axis=1,
    )
    profiles["_business_proteins"] = profiles[
        "_business_target_text"
    ].map(_protein_set)
    profiles["_business_families"] = profiles[
        "_business_target_text"
    ].map(_family_set)
    # The product form the SKU actually *is*, read from its own name rather
    # than from _business_target_text. The wider text carries the merchandising
    # sub-category too - CCF's is "POP-CORN & CHICKEN FRIES" - which would make
    # a chicken fries SKU claim popcorn as one of its own forms.
    profiles["_business_form"] = profiles.apply(
        lambda row: _family_concept_set(_safe_text(row.get("Itemname")))
        or _family_concept_set(_safe_text(row.get("Item-Cat-4"))),
        axis=1,
    )
    return profiles


def _evaluate_target(
    *,
    target: dict[str, Any],
    source_evidence: pd.DataFrame,
    competitor_pool: pd.DataFrame,
    config: CompetitorConfig,
    run_id: str,
    retained_ranks: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    # The business contract defines the uploaded ClickFlyer dump as the
    # competitor search universe. Country remains traceable source evidence,
    # but it is not a precondition for evaluating product compatibility.
    context = competitor_pool.copy()
    if context.empty:
        return pd.DataFrame(columns=COMPETITOR_LONG_COLUMNS)

    first_source = source_evidence.sort_values(
        "_business_offer_id", kind="stable"
    ).iloc[0]
    target_category = _safe_text(target.get("category"))
    target_family = _safe_text(target.get("_business_family"))
    target_proteins = target.get("_business_proteins", set())
    target_families = target.get("_business_families", set())

    status = pd.Series("UNRELATED", index=context.index, dtype="object")
    reason = pd.Series(
        "PRODUCT_FAMILY_CONFLICT", index=context.index, dtype="object"
    )
    same_category = context["category"].map(_safe_text).eq(target_category)
    reason.loc[~same_category] = "CATEGORY_CONFLICT"
    same_family = context["product_family"].map(_safe_text).eq(target_family)
    semantic_family_overlap = context["_business_families"].map(
        lambda values: bool(
            target_families
            and values
            and set(target_families).intersection(values)
        )
    )
    semantic_candidates = same_category & (
        same_family | semantic_family_overlap
    )

    protein_conflict = semantic_candidates & context[
        "_business_proteins"
    ].map(
        lambda values: bool(
            target_proteins
            and values
            and not set(target_proteins).intersection(values)
        )
    )
    family_conflict = (
        semantic_candidates
        & ~protein_conflict
        & context["_business_families"].map(
            lambda values: bool(
                target_families
                and values
                and not set(target_families).intersection(values)
            )
        )
    )
    # A competitor must sell the same product form. Chicken fries compete with
    # chicken fries, not with nuggets, strips, fillets or popcorn, even though
    # those share a protein and a category. The family sets above cannot make
    # that call: they read the whole offer name, so a bundle that merely lists
    # chicken fries among four products intersects the target and passes.
    #
    # Both sides must name a form for the rule to apply. A SKU or a Product
    # entry with no recognised form phrase is left to the checks above rather
    # than being excluded on an absence of evidence.
    target_forms = target.get("_business_form") or set()
    context_forms = (
        context["_business_form"]
        if "_business_form" in context
        else context["Product"].map(
            lambda value: _family_concept_set(_safe_text(value))
        )
    )
    form_conflict = (
        semantic_candidates
        & ~protein_conflict
        & ~family_conflict
        & context_forms.map(
            lambda values: bool(
                target_forms
                and values
                and not set(target_forms).intersection(values)
            )
        )
    )
    status.loc[protein_conflict | family_conflict | form_conflict] = (
        "HARD_CONFLICT"
    )
    reason.loc[protein_conflict] = "PROTEIN_CONFLICT"
    reason.loc[family_conflict] = "FAMILY_CONFLICT"
    reason.loc[form_conflict] = "FORM_CONFLICT"

    eligible_indices = context.index[
        semantic_candidates
        & ~protein_conflict
        & ~family_conflict
        & ~form_conflict
    ]
    raw_score = pd.Series(np.nan, index=context.index, dtype=float)
    adjusted_score = pd.Series(np.nan, index=context.index, dtype=float)
    pack_status = pd.Series(None, index=context.index, dtype="object")
    if len(eligible_indices):
        eligible = context.loc[eligible_indices]
        # Scored on the brand-stripped, transliteration-folded text rather than
        # raw match_text; see _competitor_match_text.
        candidate_text = (
            eligible["_business_match_text"].map(_safe_text).tolist()
            if "_business_match_text" in eligible
            else eligible["match_text"].map(_safe_text).tolist()
        )
        scores = (
            process.cdist(
                [_safe_text(target["_business_target_text"])],
                candidate_text,
                scorer=fuzz.token_sort_ratio,
            )
            + process.cdist(
                [_safe_text(target["_business_target_text"])],
                candidate_text,
                scorer=fuzz.token_set_ratio,
            )
        )[0] / 2.0
        flags = np.array(
            [
                pack_is_compatible(
                    target.get("master_measures", []),
                    measures,
                )
                for measures in eligible["offer_measures"]
            ],
            dtype=object,
        )
        adjusted = scores + np.where(
            flags == True,  # noqa: E712 - tri-state comparison
            3,
            np.where(flags == None, -3, 0),  # noqa: E711
        )
        raw_score.loc[eligible_indices] = scores
        adjusted_score.loc[eligible_indices] = adjusted
        pack_status.loc[eligible_indices] = flags

        known_pack_conflict = pd.Series(
            flags == False,  # noqa: E712
            index=eligible_indices,
        )
        policy_match = pd.Series(
            (scores >= config.raw_score_floor)
            & (adjusted >= config.adjusted_score_floor),
            index=eligible_indices,
        )
        conflict_indices = known_pack_conflict[
            known_pack_conflict
        ].index
        status.loc[conflict_indices] = "HARD_CONFLICT"
        reason.loc[conflict_indices] = "PACK_CONFLICT"

        passing_indices = policy_match[
            policy_match & ~known_pack_conflict
        ].index
        status.loc[passing_indices] = "MATCHED"
        reason.loc[passing_indices] = "SUPPORTED_COMPATIBLE_COMPETITOR"
        ambiguous_indices = [
            index
            for index in passing_indices
            if pack_status.loc[index] is None
        ]
        status.loc[ambiguous_indices] = "AMBIGUOUS"
        reason.loc[ambiguous_indices] = "SUPPORTED_PACK_UNVERIFIED"

        below_indices = policy_match[
            ~policy_match & ~known_pack_conflict
        ].index
        status.loc[below_indices] = "UNRELATED"
        reason.loc[
            below_indices
        ] = "BELOW_COMPETITOR_ELIGIBILITY_POLICY"

    evaluation = pd.DataFrame(
        {
            "master_sku": target["_business_master_sku"],
            "master_name": target["Itemname"],
            "master_description": target["Item Description"],
            "source_alkabeer_offer_id": first_source[
                "_business_offer_id"
            ],
            "source_alkabeer_offer_name": first_source["Offer Name"],
            "competitor_offer_id": context["_business_offer_id"],
            "competitor_offer_name": context["Offer Name"],
            "competitor_product": context["Product"],
            "competitor_brand": context["Brand Name"],
            "competitor_variant": context["Variant"],
            "competitor_pack_size": context["Base Packsize"],
            "competitor_retailer": context["Retailer Name"],
            "competitor_flyer": context["Flyer Name"],
            "competitor_offer_price": context["Offer Price"],
            "competitor_regular_price": context["Regular Price"],
            "competitor_match_score": raw_score,
            "competitor_adjusted_score": adjusted_score,
            "competitor_match_status": status,
            "competitor_match_reason": reason,
            "competitor_rank": None,
            # Every row states how it was ordered, including the ones the
            # rules settled before any model was consulted. A blank here
            # would read as "unknown" when the answer is "rules decided it".
            "competitor_lightgbm_score": None,
            "competitor_lightgbm_rank": None,
            "competitor_ranking_source": "rules",
            "run_id": run_id,
        },
        columns=COMPETITOR_LONG_COLUMNS,
    )

    if retained_ranks is not None:
        return _apply_retained_ranks(evaluation, retained_ranks)
    return evaluation


def _apply_retained_ranks(
    evaluation: pd.DataFrame, retained_ranks: Mapping[str, int]
) -> pd.DataFrame:
    """Stamp ranks and demote anything past the per-target limit.

    Extracted so the single-pass caller and :func:`_evaluate_target`'s own
    ``retained_ranks`` argument cannot diverge on what a rank means.
    """
    if evaluation.empty:
        return evaluation
    supported = evaluation["competitor_match_status"].isin(
        SUPPORTED_COMPETITOR_STATUSES
    )
    supported_offer_ids = evaluation.loc[
        supported, "competitor_offer_id"
    ].map(_safe_text)
    retained = supported & supported_offer_ids.isin(retained_ranks)
    limited = supported & ~retained
    evaluation.loc[limited, "competitor_match_status"] = "EXCLUDED"
    evaluation.loc[
        limited, "competitor_match_reason"
    ] = "BELOW_MAX_PER_TARGET_LIMIT"
    evaluation.loc[retained, "competitor_rank"] = (
        supported_offer_ids.loc[retained].map(retained_ranks)
    )
    return evaluation


#: Ranking keys for ML-scored rows are lifted into a band above every possible
#: fuzzy score (0-100), so a candidate the model has an opinion about always
#: orders ahead of one it never saw, while both keep a stable internal order.
_ML_RANK_BAND = 1000.0


@dataclass
class _MLContext:
    """Per-offer candidate shortlists and their model ordering.

    The model is a WITHIN-OFFER ranker: its group-relative features describe
    one offer against its own shortlist. Competitor discovery loops the other
    way round - one master SKU against many offers - so a shortlist is built
    here per competitor offer, using the same category-gated RapidFuzz
    generator and the same top-K the model was trained against, and cached so
    an offer that is a candidate for several targets is featurised once.
    """

    reranker: Any
    generator: Any
    master_lookup: Mapping[str, Any]
    top_k: int
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    offers_ranked: int = 0
    pairs_scored: int = 0

    def warm(self, offers: list[tuple[str, Any]]) -> None:
        """Build and score every missing shortlist in one batch.

        Retrieval and inference are both batch operations; driving them one
        offer at a time spends nearly all of its time on per-call overhead
        rather than on the work. Measured on an 8k-row slice, per-offer calls
        cost 17.7x the rules alone - almost none of it in the model.
        """
        pending = [
            (offer_id, offer_row)
            for offer_id, offer_row in offers
            if offer_id not in self.cache
        ]
        if not pending:
            return
        try:
            shortlists = self.generator.generate_candidates_batch(
                pd.DataFrame([row for _, row in pending]), top_k=self.top_k
            )
            batch: list[tuple[str, Any, list[Any]]] = []
            for (offer_id, offer_row), candidates in zip(pending, shortlists):
                master_rows = [
                    self.master_lookup[candidate.itemcode]
                    for candidate in candidates
                    if candidate.itemcode in self.master_lookup
                ]
                batch.append((offer_id, offer_row, master_rows))
            ranked_by_offer = self.reranker.rank_many(batch)
        except Exception:
            # A shortlist that cannot be built is a missing opinion, not a
            # failed run: the rows keep their rule ordering.
            LOGGER.warning(
                "Competitor re-ranking failed for a batch; rule order kept",
                exc_info=True,
            )
            ranked_by_offer = {}
        for offer_id, _ in pending:
            ranked = ranked_by_offer.get(offer_id, {})
            self.cache[offer_id] = ranked
            self.offers_ranked += 1
            self.pairs_scored += len(ranked)

    def ranks_for_offer(self, offer_id: str, offer_row: Any) -> dict[str, Any]:
        cached = self.cache.get(offer_id)
        if cached is not None:
            return cached
        self.warm([(offer_id, offer_row)])
        return self.cache.get(offer_id, {})


def _apply_ml_reranking(
    frame: pd.DataFrame,
    *,
    target: Mapping[str, Any],
    offer_lookup: Mapping[str, Any],
    ml_context: "_MLContext | None",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Attach the model's opinion to rows the rules already admitted.

    Never changes a status and never adds a row. The only thing it can change
    is ``_ranking_key``, which decides display order and - when
    ``max_per_target`` is set - which rows survive the cut.
    """
    stats = {"offers_ranked": 0, "pairs_scored": 0, "rows_with_model_score": 0}
    if frame.empty:
        return frame, stats
    frame = frame.copy()
    fuzzy = pd.to_numeric(
        frame["competitor_match_score"], errors="coerce"
    ).fillna(float("-inf"))
    if ml_context is None:
        frame["competitor_lightgbm_score"] = None
        frame["competitor_lightgbm_rank"] = None
        frame["competitor_ranking_source"] = "rules"
        frame["_ranking_key"] = fuzzy
        return frame, stats

    master_sku = _safe_text(target.get("_business_master_sku"))
    scores: list[float | None] = []
    ranks: list[int | None] = []
    before = (ml_context.offers_ranked, ml_context.pairs_scored)
    # Warm the whole target's offers at once. Offers already seen for an
    # earlier target are skipped, so the cost falls on distinct offers.
    ml_context.warm(
        [
            (offer_id, offer_lookup[offer_id])
            for offer_id in dict.fromkeys(
                frame["competitor_offer_id"].map(_safe_text)
            )
            if offer_id in offer_lookup
        ]
    )
    for offer_id in frame["competitor_offer_id"].map(_safe_text):
        offer_row = offer_lookup.get(offer_id)
        if offer_row is None:
            scores.append(None)
            ranks.append(None)
            continue
        ranked = ml_context.ranks_for_offer(offer_id, offer_row)
        entry = ranked.get(master_sku)
        scores.append(None if entry is None else float(entry.raw_margin))
        ranks.append(None if entry is None else int(entry.rank))
    frame["competitor_lightgbm_score"] = scores
    frame["competitor_lightgbm_rank"] = ranks
    has_score = pd.Series([score is not None for score in scores], index=frame.index)
    frame["competitor_ranking_source"] = np.where(has_score, "lightgbm", "rules")
    margins = pd.Series(
        [0.0 if score is None else float(score) for score in scores],
        index=frame.index,
    )
    frame["_ranking_key"] = np.where(
        has_score, _ML_RANK_BAND + margins, fuzzy
    )
    stats["offers_ranked"] = ml_context.offers_ranked - before[0]
    stats["pairs_scored"] = ml_context.pairs_scored - before[1]
    stats["rows_with_model_score"] = int(has_score.sum())
    return frame, stats


def _accumulate(totals: dict[str, float], additions: Mapping[str, float]) -> None:
    for key, value in additions.items():
        totals[key] = totals.get(key, 0) + value


def _build_ml_context(
    reranker: Any,
    product_master: pd.DataFrame,
    config: CompetitorConfig | None = None,
) -> "_MLContext | None":
    """Pair a re-ranker with the production shortlist generator.

    The shortlist comes from :class:`CandidateGenerator` - the same
    category-gated RapidFuzz retrieval the own-brand pipeline uses and the
    model was trained against - so the candidate groups the model sees here
    have the shape it was fitted on. Top-K defaults to the package's own
    ``retrieval_k`` rather than a number chosen here.
    """
    try:
        from sku_mapping.matching.candidate_generator import CandidateGenerator

        master = product_master.reset_index(drop=True)
        generator = CandidateGenerator(master)
        master_lookup = {
            _safe_text(row["Itemcode"]): row for _, row in master.iterrows()
        }
        # Configured K wins; 0 defers to the package's own retrieval_k, which
        # is the group size the model was fitted against. Falling back to 20
        # only matters for a package that declares none.
        configured = int(getattr(config, "ml_shortlist_top_k", 0) or 0)
        top_k = configured or int(getattr(reranker, "retrieval_k", 0) or 0) or 20
        return _MLContext(
            reranker=reranker,
            generator=generator,
            master_lookup=master_lookup,
            top_k=top_k,
        )
    except Exception:
        LOGGER.warning(
            "Could not build the competitor shortlist generator", exc_info=True
        )
        return None


def _retained_ranks(
    supported_rows: list[tuple[float, str]],
    max_per_target: int,
) -> dict[str, int]:
    ordered = sorted(supported_rows, key=lambda item: (-item[0], item[1]))
    # 0 means no limit: keep every competitor that cleared the score floors,
    # still ranked best-first. Anything dropped here is reported as
    # BELOW_MAX_PER_TARGET_LIMIT, so a truncated list is never silent.
    if max_per_target > 0:
        ordered = ordered[:max_per_target]
    return {
        offer_id: rank
        for rank, (_, offer_id) in enumerate(ordered, start=1)
    }


def _append_audit_frame(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    frame.loc[:, COMPETITOR_LONG_COLUMNS].to_csv(
        path,
        mode="a",
        header=False,
        index=False,
        encoding="utf-8",
    )


def _aggregate_target(
    *,
    target: dict[str, Any],
    source_evidence: pd.DataFrame,
    evaluations: pd.DataFrame,
    reason_counts: Mapping[str, int],
    source_count: int,
    source_entity_ids: list[str],
    run_id: str,
) -> dict[str, object]:
    sources = source_evidence.sort_values(
        "_business_offer_id", kind="stable"
    ).drop_duplicates("_business_offer_id", keep="first")
    supported = evaluations[
        evaluations["competitor_match_status"].isin(
            SUPPORTED_COMPETITOR_STATUSES
        )
    ].copy()
    supported = supported.sort_values(
        ["competitor_rank", "competitor_offer_id"],
        kind="stable",
    )
    # The pool now carries one row per (offer, variant), so a single offer whose
    # variants both pass would otherwise be listed twice. Sorted by rank above,
    # so keep="first" retains that offer's best-scoring variant.
    supported = supported.drop_duplicates("competitor_offer_id", keep="first")
    count = int(len(supported))
    return {
        "master_sku": target["_business_master_sku"],
        "master_name": target["Itemname"],
        "master_description": target["Item Description"],
        "source_alkabeer_offer_ids": _json_list(
            sources["_business_offer_id"].tolist()
        ),
        "source_entity_ids": _json_list(source_entity_ids),
        "source_alkabeer_offer_names": _json_list(
            sources["Offer Name"].tolist()
        ),
        "competitor_count": count,
        "competitor_brand_names": _json_list(
            supported["competitor_brand"].tolist()
        ),
        "competitor_offer_ids": _json_list(
            supported["competitor_offer_id"].tolist()
        ),
        "competitor_offer_names": _json_list(
            supported["competitor_offer_name"].tolist()
        ),
        "competitor_products": _json_list(
            supported["competitor_product"].tolist()
        ),
        "competitor_variants": _json_list(
            supported["competitor_variant"].tolist()
        ),
        "competitor_pack_sizes": _json_list(
            supported["competitor_pack_size"].tolist()
        ),
        "competitor_retailers": _json_list(
            supported["competitor_retailer"].tolist()
        ),
        "competitor_flyers": _json_list(
            supported["competitor_flyer"].tolist()
        ),
        "competitor_offer_prices": _json_list(
            supported["competitor_offer_price"].tolist()
        ),
        "competitor_regular_prices": _json_list(
            supported["competitor_regular_price"].tolist()
        ),
        "competitor_status": (
            "COMPETITORS_FOUND" if count else "NO_COMPETITOR_FOUND"
        ),
        "competitor_reason": (
            f"{count} supported competitor offer(s) from uploaded dump"
            if count
            else _target_reason(reason_counts, source_count)
        ),
        "run_id": run_id,
    }


#: Columns every competitor-pool row must carry to be evaluated.
COMPETITOR_POOL_COLUMNS = {
    "offer_group_id",
    "is_own",
    "Country",
    "Offer Name",
    "Product",
    "Brand Name",
    "Variant",
    "Base Packsize",
    "Retailer Name",
    "Flyer Name",
    "Offer Price",
    "Regular Price",
    "category",
    "product_family",
    "match_text",
    "offer_measures",
}


def discover_competitors(
    prepared_offers: pd.DataFrame,
    product_master: pd.DataFrame,
    sku_mapping: pd.DataFrame,
    *,
    config: CompetitorConfig,
    run_id: str,
    audit_path: str | Path | None = None,
    progress: CompetitorProgressCallback | None = None,
    competitor_offers: pd.DataFrame | None = None,
    reranker: Any | None = None,
    adjudicator: Any | None = None,
) -> CompetitorDiscoveryResult:
    """Discover exact dump competitors for distinct mapped Master SKUs.

    ``sku_mapping`` is the business mapping table, not the lifecycle decision
    table.  Consequently a visible manual-review proposal is a valid target,
    while an empty/no-candidate mapping is not.

    ``competitor_offers`` supplies the competitor search pool when the caller
    holds a finer-grained frame than ``prepared_offers``.  A ClickFlyer offer
    is repeated once per variant it covers, and a caller that has already
    collapsed the dump to one row per offer identity would otherwise present
    only the first variant - in practice the ``No Variant`` row - hiding the
    variant that carries the protein evidence.  Own-brand source evidence is
    always taken from ``prepared_offers`` so mapping traceability is unchanged.

    ``reranker`` is optional. When supplied it reorders candidates the rules
    have already admitted - it cannot admit, reject, or override a conflict.
    When absent (the default) the rules order the list exactly as before, so
    the pre-ML behaviour is what a caller gets by not passing anything.
    """

    _require_columns(
        prepared_offers,
        COMPETITOR_POOL_COLUMNS,
        label="Prepared ClickFlyer offers",
    )
    if competitor_offers is not None:
        _require_columns(
            competitor_offers,
            COMPETITOR_POOL_COLUMNS,
            label="Competitor search pool",
        )
    _require_columns(
        product_master,
        {
            "Itemcode",
            "Itemname",
            "Item-Cat-4",
            "Item Description",
            "category",
            "master_measures",
        },
        label="Product Master",
    )
    _require_columns(
        sku_mapping,
        {
            "source_offer_id",
            "source_offer_name",
            "matched_master_sku",
        },
        label="SKU mapping export",
    )
    if not _safe_text(run_id):
        raise ValueError("run_id is required for competitor discovery")

    canonical = _canonical_sources(prepared_offers)
    competitor_pool = _canonical_competitors(
        prepared_offers if competitor_offers is None else competitor_offers
    )
    competitor_pool["_business_match_text"] = _competitor_match_text(
        competitor_pool
    )
    competitor_pool["_business_proteins"] = (
        competitor_pool["Offer Name"].map(_safe_text)
        + " "
        + competitor_pool["Product"].map(_safe_text)
    ).map(_protein_set)
    competitor_pool["_business_families"] = (
        competitor_pool["Offer Name"].map(_safe_text)
        + " "
        + competitor_pool["Product"].map(_safe_text)
    ).map(_family_set)
    # Form is read from ``Product`` alone - the offer's primary item - and not
    # from the offer name. A combo flyer ("Chicken Nuggets/Popcorn/Fries/
    # Fillet 400gm x2") names several products, so the offer name reports every
    # form in the bundle and cannot say which one the offer is for.
    competitor_pool["_business_form"] = competitor_pool["Product"].map(
        lambda value: _family_concept_set(_safe_text(value))
    )
    category_indices = {
        _safe_text(category): list(indices)
        for category, indices in competitor_pool.groupby(
            "category", sort=False
        ).groups.items()
    }

    # Model wiring. Built once and shared by every target: an offer that is a
    # candidate for several master SKUs is featurised on its first appearance
    # and read from the cache afterwards, which is what keeps the ML cost
    # proportional to distinct supported offers rather than to relationships.
    ml_context: _MLContext | None = None
    ml_diagnostics: dict[str, float] = {}
    if reranker is not None:
        ml_context = _build_ml_context(reranker, product_master, config)
        if ml_context is None:
            LOGGER.warning(
                "Competitor re-ranker supplied but no shortlist generator "
                "could be built; continuing on rule ordering"
            )
    offer_lookup = {
        _safe_text(row["_business_offer_id"]): row
        for _, row in competitor_pool.iterrows()
    } if ml_context is not None else {}

    mappings = sku_mapping.copy()
    mappings["_business_master_sku"] = mappings[
        "matched_master_sku"
    ].map(_safe_text)
    mappings = mappings[mappings["_business_master_sku"].ne("")]
    target_codes = (
        mappings["_business_master_sku"]
        .drop_duplicates(keep="first")
        .tolist()
    )
    profiles = _master_profiles(product_master)
    target_lookup = {
        _safe_text(record["_business_master_sku"]): record
        for record in profiles.to_dict(orient="records")
    }
    source_lookup = canonical.set_index("_business_offer_id", drop=False)

    aggregate_by_code: dict[str, dict[str, object]] = {}
    evaluation_frames: list[pd.DataFrame] = []
    audit_output = Path(audit_path) if audit_path is not None else None
    if audit_output is not None:
        audit_output.parent.mkdir(parents=True, exist_ok=True)
        if audit_output.exists():
            raise FileExistsError(
                f"Refusing to overwrite competitor audit: {audit_output}"
            )
        pd.DataFrame(columns=COMPETITOR_LONG_COLUMNS).to_csv(
            audit_output, index=False, encoding="utf-8-sig"
        )

    total_relationships = len(target_codes) * len(competitor_pool)
    processed_relationships = 0
    supported_row_count = 0
    supported_offer_ids: set[str] = set()
    all_reason_counts: dict[str, int] = {}
    processing_codes = (
        sorted(target_codes) if audit_output is not None else target_codes
    )
    if progress is not None:
        progress(
            0,
            len(processing_codes),
            0,
            total_relationships,
            "",
        )
    for target_position, code in enumerate(processing_codes, start=1):
        target = target_lookup.get(code)
        if target is None:
            raise ValueError(
                f"Mapped target Master SKU is absent from Product Master: {code}"
            )
        mapped_sources = mappings[
            mappings["_business_master_sku"].eq(code)
        ]["source_offer_id"].map(_safe_text)
        mapped_entity_ids = (
            mappings.loc[
                mappings["_business_master_sku"].eq(code), "entity_id"
            ].map(_safe_text)
            if "entity_id" in mappings.columns
            else mapped_sources
        )
        source_ids = [
            source_id
            for source_id in mapped_sources.tolist()
            if source_id in source_lookup.index
        ]
        source_evidence = source_lookup.loc[source_ids]
        if isinstance(source_evidence, pd.Series):
            source_evidence = source_evidence.to_frame().T
        source_evidence = source_evidence.reset_index(drop=True)
        if source_evidence.empty:
            raise ValueError(
                f"Mapped target {code} has no traceable own-brand source offer"
            )
        target_category = _safe_text(target.get("category"))
        target_pool = competitor_pool.loc[
            category_indices.get(target_category, [])
        ]
        omitted_category_count = len(competitor_pool) - len(target_pool)
        target_reason_counts: dict[str, int] = {}
        if omitted_category_count:
            target_reason_counts["CATEGORY_CONFLICT"] = (
                omitted_category_count
            )
        supported_rows: list[tuple[float, str]] = []

        # ONE evaluation pass. Every relationship used to be scored twice -
        # once to collect the scores that decide ranks, once to emit rows -
        # which doubled the cost of the most expensive stage in the run. Only
        # supported rows can be changed by a rank, so unsupported rows are
        # final the moment they are produced and go straight out; the far
        # smaller supported set is held back until ranks are known.
        settled_frames: list[pd.DataFrame] = []
        pending_supported: list[pd.DataFrame] = []
        for start in range(
            0, len(target_pool), COMPETITOR_EVALUATION_CHUNK_SIZE
        ):
            chunk = target_pool.iloc[
                start : start + COMPETITOR_EVALUATION_CHUNK_SIZE
            ]
            evaluated = _evaluate_target(
                target=target,
                source_evidence=source_evidence,
                competitor_pool=chunk,
                config=config,
                run_id=run_id,
            )
            supported = evaluated["competitor_match_status"].isin(
                SUPPORTED_COMPETITOR_STATUSES
            )
            supported_rows.extend(
                (
                    float(score),
                    _safe_text(offer_id),
                )
                for score, offer_id in zip(
                    pd.to_numeric(
                        evaluated.loc[
                            supported, "competitor_match_score"
                        ],
                        errors="coerce",
                    ).fillna(float("-inf")),
                    evaluated.loc[supported, "competitor_offer_id"],
                )
            )
            for reason, count in (
                evaluated.loc[
                    ~supported, "competitor_match_reason"
                ].value_counts()
            ).items():
                target_reason_counts[str(reason)] = (
                    target_reason_counts.get(str(reason), 0) + int(count)
                )
            if supported.any():
                pending_supported.append(evaluated.loc[supported])
            settled = evaluated.loc[~supported]
            if not settled.empty:
                if audit_output is None:
                    settled_frames.append(settled)
                else:
                    _append_audit_frame(settled, audit_output)
            # Emit a sub-SKU heartbeat so the dashboard bar moves even
            # when a single target SKU has a very large competitor pool.
            if progress is not None:
                chunk_relationships = processed_relationships + start + len(chunk)
                progress(
                    target_position,
                    len(processing_codes),
                    min(chunk_relationships, total_relationships),
                    total_relationships,
                    code,
                )

        supported_frame = (
            pd.concat(pending_supported, ignore_index=True)
            if pending_supported
            else pd.DataFrame(columns=COMPETITOR_LONG_COLUMNS)
        )
        # The learned signal enters HERE and only here: it reorders the rows
        # the rules already admitted. It cannot add a row, and the ordering it
        # produces feeds the same max_per_target cut the fuzzy order fed.
        supported_frame, ml_stats = _apply_ml_reranking(
            supported_frame,
            target=target,
            offer_lookup=offer_lookup,
            ml_context=ml_context,
        )
        _accumulate(ml_diagnostics, ml_stats)
        if ml_context is not None and not supported_frame.empty:
            supported_rows = [
                (float(score), _safe_text(offer_id))
                for score, offer_id in zip(
                    supported_frame["_ranking_key"],
                    supported_frame["competitor_offer_id"],
                )
            ]

        ranks = _retained_ranks(supported_rows, config.max_per_target)
        limited_count = max(0, len(supported_rows) - len(ranks))
        if limited_count:
            target_reason_counts["BELOW_MAX_PER_TARGET_LIMIT"] = (
                target_reason_counts.get(
                    "BELOW_MAX_PER_TARGET_LIMIT", 0
                )
                + limited_count
            )

        supported_frame = _apply_retained_ranks(supported_frame, ranks)
        supported_frame = supported_frame.drop(
            columns=["_ranking_key"], errors="ignore"
        )
        retained = supported_frame[
            supported_frame["competitor_match_status"].isin(
                SUPPORTED_COMPETITOR_STATUSES
            )
        ]
        if not retained.empty:
            supported_row_count += len(retained)
            supported_offer_ids.update(
                retained["competitor_offer_id"].map(_safe_text)
            )
        if not supported_frame.empty:
            if audit_output is None:
                settled_frames.append(supported_frame)
            else:
                _append_audit_frame(supported_frame, audit_output)

        retained_frames = [retained] if not retained.empty else []
        target_frames = settled_frames
        target_evaluations = (
            pd.concat(target_frames, ignore_index=True)
            if target_frames
            else pd.DataFrame(columns=COMPETITOR_LONG_COLUMNS)
        )
        retained_evaluations = (
            pd.concat(retained_frames, ignore_index=True)
            if retained_frames
            else pd.DataFrame(columns=COMPETITOR_LONG_COLUMNS)
        )
        if audit_output is None:
            evaluation_frames.append(target_evaluations)
        aggregate_by_code[code] = _aggregate_target(
            target=target,
            source_evidence=source_evidence,
            evaluations=retained_evaluations,
            reason_counts=target_reason_counts,
            source_count=len(competitor_pool),
            source_entity_ids=[
                value
                for value in mapped_entity_ids.drop_duplicates().tolist()
                if value
            ],
            run_id=run_id,
        )
        for reason, count in target_reason_counts.items():
            all_reason_counts[reason] = (
                all_reason_counts.get(reason, 0) + int(count)
            )
        processed_relationships += len(competitor_pool)
        if progress is not None:
            progress(
                target_position,
                len(processing_codes),
                processed_relationships,
                total_relationships,
                code,
            )

    export = pd.DataFrame(
        [aggregate_by_code[code] for code in target_codes],
        columns=COMPETITOR_EXPORT_COLUMNS,
    )
    long_format = (
        pd.concat(evaluation_frames, ignore_index=True)
        if evaluation_frames
        else pd.DataFrame(columns=COMPETITOR_LONG_COLUMNS)
    )
    if not long_format.empty:
        long_format["_supported_sort"] = ~long_format[
            "competitor_match_status"
        ].isin(SUPPORTED_COMPETITOR_STATUSES)
        long_format["_score_sort"] = pd.to_numeric(
            long_format["competitor_match_score"], errors="coerce"
        )
        long_format = (
            long_format.sort_values(
                [
                    "master_sku",
                    "_supported_sort",
                    "_score_sort",
                    "competitor_offer_id",
                ],
                ascending=[True, True, False, True],
                kind="stable",
                na_position="last",
            )
            .drop(columns=["_supported_sort", "_score_sort"])
            .reset_index(drop=True)
        )

    # The terminal stage. Ranking above decides ORDER; this decides OUTCOME,
    # grouped by competitor offer rather than by target, because "which SKU
    # does this rival product compete with" has one answer per offer. Off by
    # configuration leaves every row undecided and the frame unchanged.
    decision_stats: dict[str, int] = {}
    if getattr(config, "automatic_decisions_enabled", False):
        from sku_mapping.competitors.decisions import apply_automatic_decisions

        # The frame is built against COMPETITOR_LONG_COLUMNS, which now carries
        # the decision columns so the audit contract is config-independent.
        # apply_automatic_decisions appends its own, so drop the placeholders
        # first or the frame ends up with each column twice.
        long_format = long_format.drop(
            columns=[
                column
                for column in (
                    "competitor_decision",
                    "competitor_decision_reason",
                    "competitor_decision_source",
                )
                if column in long_format.columns
            ]
        )
        long_format, decision_stats = apply_automatic_decisions(
            long_format,
            clear_margin=float(getattr(config, "clear_margin_threshold", 0.0)),
            clear_gap=float(getattr(config, "clear_gap_threshold", 2.0)),
            adjudicator=adjudicator,
            max_adjudicated_candidates=int(
                getattr(config, "llm_max_candidates", 5)
            ),
        )

    # Guarantee the audit shape regardless of whether decisions ran, so the
    # export validator sees one contract instead of two.
    for column in ("competitor_decision", "competitor_decision_reason",
                   "competitor_decision_source"):
        if column not in long_format.columns:
            long_format[column] = None

    source_offer_ids = set(
        competitor_pool["_business_offer_id"].map(_safe_text)
    )
    diagnostics: dict[str, object] = {
        "target_master_sku_count": int(len(target_codes)),
        "source_competitor_offer_count": int(len(competitor_pool)),
        "source_competitor_offers_evaluated": int(len(competitor_pool)),
        "target_offer_relationships_evaluated": int(total_relationships),
        "detailed_relationship_rows": int(
            len(long_format)
            if audit_output is None
            else sum(
                len(category_indices.get(
                    _safe_text(target_lookup[code].get("category")), []
                ))
                for code in target_codes
            )
        ),
        **{
            f"decision_{key}": int(value)
            for key, value in decision_stats.items()
        },
        "matched_competitor_rows": int(supported_row_count),
        "matched_competitor_offer_count": int(
            len(supported_offer_ids)
        ),
        "no_match_target_count": int(
            export["competitor_status"].eq("NO_COMPETITOR_FOUND").sum()
            if not export.empty
            else 0
        ),
        "excluded_relationship_count": int(
            total_relationships - supported_row_count
        ),
        "excluded_source_offer_count": int(
            len(source_offer_ids - supported_offer_ids)
        ),
        "exclusion_reason_counts": {
            str(reason): int(count)
            for reason, count in sorted(all_reason_counts.items())
        },
        # Observability for the learned layer. ``enabled`` false means the run
        # was pure rules, which is also what every field below reads as zero.
        "ml_reranking": {
            "enabled": bool(ml_context is not None),
            "model_id": (
                str(getattr(reranker, "model_id", "")) if ml_context else ""
            ),
            "shortlist_top_k": int(ml_context.top_k) if ml_context else 0,
            "offers_ranked": int(ml_diagnostics.get("offers_ranked", 0)),
            "pairs_scored": int(ml_diagnostics.get("pairs_scored", 0)),
            "rows_with_model_score": int(
                ml_diagnostics.get("rows_with_model_score", 0)
            ),
            "score_semantics": "raw_margin_within_offer_shortlist",
            "used_for": "ranking_only",
        },
    }
    if export["master_sku"].duplicated().any():
        raise ValueError("Competitor aggregate contains duplicate Master SKUs")
    if len(export) != len(target_codes):
        raise ValueError(
            "Competitor row count does not equal distinct mapped Master SKU count"
        )
    return CompetitorDiscoveryResult(
        export=export,
        # The canonical columns in their canonical order. DECISION_COLUMNS are
        # part of that tuple now and are always materialised above, so the
        # audit has one shape whether or not the decision layer ran.
        long_format=long_format.loc[:, list(COMPETITOR_LONG_COLUMNS)],
        long_format_path=audit_output,
        eligible_target_count=int(len(target_codes)),
        diagnostics=diagnostics,
    )
