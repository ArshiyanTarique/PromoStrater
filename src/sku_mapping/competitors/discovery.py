"""Master-SKU-centric competitor discovery over one ClickFlyer dump.

The internal long-form table records every evaluated relationship in the
uploaded dump.  The user-facing aggregate is derived only from supported rows
in that table and therefore cannot develop misaligned lists.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from sku_mapping.config import CompetitorConfig
from sku_mapping.data.preprocessing import normalize_product_family
from sku_mapping.features.measurement_features import pack_is_compatible
from sku_mapping.features.semantic_features import _family_set, _protein_set

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
    "run_id",
)

SUPPORTED_COMPETITOR_STATUSES = frozenset({"MATCHED", "AMBIGUOUS"})
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


#: Transliterations of the same product word. ClickFlyer carries retailer copy
#: verbatim, so the same item appears as "Sambosa" or "Samosa" depending on who
#: wrote the flyer, and a token-based ratio treats those as unrelated words.
#: Applied to competitor scoring text only - own-brand mapping is untouched.
_COMPETITOR_TOKEN_ALIASES = {
    "sambosa": "samosa",
    "sambosas": "samosa",
    "samboosa": "samosa",
    "sambousa": "samosa",
    "samosas": "samosa",
}


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
    texts = pool["match_text"].map(_safe_text).str.lower()
    brands = pool["Brand Name"].map(_safe_text).str.lower()
    products = pool["Product"].map(_safe_text).str.lower()

    cleaned: list[str] = []
    for text, brand, product in zip(texts, brands, products):
        brand_tokens = set(brand.replace("-", " ").split())
        protected = set(product.replace("-", " ").split())
        tokens = [
            _COMPETITOR_TOKEN_ALIASES.get(token, token)
            for token in text.split()
            if token not in brand_tokens or token in protected
        ]
        cleaned.append(" ".join(tokens))
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
    status.loc[protein_conflict | family_conflict] = "HARD_CONFLICT"
    reason.loc[protein_conflict] = "PROTEIN_CONFLICT"
    reason.loc[family_conflict] = "FAMILY_CONFLICT"

    eligible_indices = context.index[
        semantic_candidates & ~protein_conflict & ~family_conflict
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
            "run_id": run_id,
        },
        columns=COMPETITOR_LONG_COLUMNS,
    )

    if retained_ranks is not None:
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
    category_indices = {
        _safe_text(category): list(indices)
        for category, indices in competitor_pool.groupby(
            "category", sort=False
        ).groups.items()
    }

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

        for start in range(
            0, len(target_pool), COMPETITOR_EVALUATION_CHUNK_SIZE
        ):
            chunk = target_pool.iloc[
                start : start + COMPETITOR_EVALUATION_CHUNK_SIZE
            ]
            initial = _evaluate_target(
                target=target,
                source_evidence=source_evidence,
                competitor_pool=chunk,
                config=config,
                run_id=run_id,
            )
            supported = initial["competitor_match_status"].isin(
                SUPPORTED_COMPETITOR_STATUSES
            )
            supported_rows.extend(
                (
                    float(score),
                    _safe_text(offer_id),
                )
                for score, offer_id in zip(
                    pd.to_numeric(
                        initial.loc[
                            supported, "competitor_match_score"
                        ],
                        errors="coerce",
                    ).fillna(float("-inf")),
                    initial.loc[supported, "competitor_offer_id"],
                )
            )
            for reason, count in (
                initial.loc[
                    ~supported, "competitor_match_reason"
                ].value_counts()
            ).items():
                target_reason_counts[str(reason)] = (
                    target_reason_counts.get(str(reason), 0) + int(count)
                )

        ranks = _retained_ranks(supported_rows, config.max_per_target)
        limited_count = max(0, len(supported_rows) - len(ranks))
        if limited_count:
            target_reason_counts["BELOW_MAX_PER_TARGET_LIMIT"] = (
                target_reason_counts.get(
                    "BELOW_MAX_PER_TARGET_LIMIT", 0
                )
                + limited_count
            )

        retained_frames: list[pd.DataFrame] = []
        target_frames: list[pd.DataFrame] = []
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
                retained_ranks=ranks,
            )
            retained = evaluated[
                evaluated["competitor_match_status"].isin(
                    SUPPORTED_COMPETITOR_STATUSES
                )
            ]
            if not retained.empty:
                retained_frames.append(retained)
                supported_row_count += len(retained)
                supported_offer_ids.update(
                    retained["competitor_offer_id"].map(_safe_text)
                )
            if audit_output is None:
                target_frames.append(evaluated)
            else:
                _append_audit_frame(evaluated, audit_output)
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
    }
    if export["master_sku"].duplicated().any():
        raise ValueError("Competitor aggregate contains duplicate Master SKUs")
    if len(export) != len(target_codes):
        raise ValueError(
            "Competitor row count does not equal distinct mapped Master SKU count"
        )
    return CompetitorDiscoveryResult(
        export=export,
        long_format=long_format.loc[:, COMPETITOR_LONG_COLUMNS],
        long_format_path=audit_output,
        eligible_target_count=int(len(target_codes)),
        diagnostics=diagnostics,
    )
