"""Competitor ML re-ranking: transfer safety, ordering, and fallback.

The model here is the own-brand ranker being borrowed. These tests pin the
properties that make borrowing it safe - it may reorder, it may not decide -
rather than any particular score, which would only encode today's package.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sku_mapping.competitors.discovery import (
    COMPETITOR_LONG_COLUMNS,
    SUPPORTED_COMPETITOR_STATUSES,
    discover_competitors,
)
from sku_mapping.competitors.reranker import (
    CompetitorReranker,
    CompetitorRerankerError,
    load_competitor_reranker,
)
from sku_mapping.competitors.text_normalisation import strip_competitor_brand
from sku_mapping.config import load_config
from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.data.preprocessing import (
    preprocess_clickflyer,
    preprocess_product_master,
)
from sku_mapping.exports.business_outputs import build_sku_mapping_export


# --------------------------------------------------------------------------
# Fixtures: a small catalogue with two forms and two proteins, so conflicts
# are real rather than asserted.
# --------------------------------------------------------------------------
def _raw_offers() -> pd.DataFrame:
    common = {
        "Country": "KSA",
        "Retailer Name": "Retailer",
        "Flyer Name": "Flyer",
        "Variant": "",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "offerid": "own-1",
                "Offer Name": "Al Kabeer Chicken Samosa 240g",
                "Offer Price": 10,
                "Regular Price": 12,
                "Brand Name": "Al Kabeer",
                "Product": "Samosa-Frozen",
                "Base Packsize": "240 g",
            },
            {
                **common,
                "offerid": "comp-a",
                "Offer Name": "Alpha Chicken Samosa 240g",
                "Offer Price": 9,
                "Regular Price": 11,
                "Brand Name": "Alpha",
                "Product": "Samosa-Frozen",
                "Base Packsize": "240 g",
            },
            {
                **common,
                "offerid": "comp-b",
                "Offer Name": "Beta Chicken Samosa 240g",
                "Offer Price": 8,
                "Regular Price": 10,
                "Brand Name": "Beta",
                "Product": "Samosa-Frozen",
                "Base Packsize": "240 g",
            },
            {
                **common,
                "offerid": "comp-beef",
                "Offer Name": "Gamma Beef Samosa 240g",
                "Offer Price": 8,
                "Regular Price": 10,
                "Brand Name": "Gamma",
                "Product": "Samosa-Frozen",
                "Base Packsize": "240 g",
            },
            {
                **common,
                "offerid": "comp-nugget",
                "Offer Name": "Delta Chicken Nuggets 240g",
                "Offer Price": 8,
                "Regular Price": 10,
                "Brand Name": "Delta",
                "Product": "Chicken Nuggets-Frozen",
                "Base Packsize": "240 g",
            },
            {
                **common,
                "offerid": "comp-bigpack",
                "Offer Name": "Epsilon Chicken Samosa 900g",
                "Offer Price": 20,
                "Regular Price": 24,
                "Brand Name": "Epsilon",
                "Product": "Samosa-Frozen",
                "Base Packsize": "900 g",
            },
        ]
    )


def _master() -> pd.DataFrame:
    return preprocess_product_master(
        pd.DataFrame(
            [
                {
                    "Itemcode": "SKU-CHICKEN-SAMOSA",
                    "Itemname": "Chicken Samosa",
                    "Item-Cat-2": "Dough",
                    "Item-Cat-4": "Chicken Samosa",
                    "Item Description": "Frozen Chicken Samosa 240g",
                    "Item-Spec": "240 GRM x 12 Pkts",
                },
                {
                    "Itemcode": "SKU-CHICKEN-NUGGETS",
                    "Itemname": "Chicken Nuggets",
                    "Item-Cat-2": "Chicken",
                    "Item-Cat-4": "Chicken Nuggets",
                    "Item Description": "Frozen Chicken Nuggets 240g",
                    "Item-Spec": "240 GRM x 12 Pkts",
                },
            ]
        )
    )


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prepared = preprocess_clickflyer(_raw_offers())
    prepared["offer_group_id"] = prepared["offerid"]
    decisions = pd.DataFrame(
        [
            {
                "offer_id": "own-1",
                "offer_name": "Al Kabeer Chicken Samosa 240g",
                "matched_master_sku": "SKU-CHICKEN-SAMOSA",
                "decision": "AUTO_ACCEPT",
                "score": 0.99,
                "reason_codes": "[]",
            }
        ]
    )
    return prepared, _master(), decisions


def _run(reranker=None) -> pd.DataFrame:
    prepared, master, decisions = _frames()
    mapping = build_sku_mapping_export(
        prepared, master, decisions, run_id="rr-run"
    )
    return discover_competitors(
        prepared,
        master,
        mapping,
        config=load_config("config/default.yaml").competitors,
        run_id="rr-run",
        reranker=reranker,
    )


class _StubReranker:
    """Deterministic stand-in with the production re-ranker's contract.

    Ordering is by offer id, which is independent of every rule signal, so a
    test can tell whether ordering came from the model or from the fuzzy score
    without depending on the real package being present.
    """

    model_id = "stub-model"
    retrieval_k = 20

    def __init__(self) -> None:
        self.calls: list[str] = []

    def rank(self, offer_row, master_rows):
        from sku_mapping.competitors.reranker import RerankedCandidate

        offer_id = str(offer_row.get("_business_offer_id", ""))
        self.calls.append(offer_id)
        codes = [str(row["Itemcode"]) for row in master_rows]
        return {
            code: RerankedCandidate(
                itemcode=code,
                raw_margin=-float(len(offer_id)) - index,
                rank=index + 1,
            )
            for index, code in enumerate(sorted(codes))
        }

    def rank_many(self, offers):
        """Batch entry point discovery actually uses.

        Delegating to :meth:`rank` keeps the double honest: whatever the batch
        path does, it must agree with scoring the offers one at a time.
        """
        return {
            offer_id: self.rank(offer_row, master_rows)
            for offer_id, offer_row, master_rows in offers
        }


# --------------------------------------------------------------------------
# B. Competitor feature parity
# --------------------------------------------------------------------------
def test_reranker_scores_competitor_pairs_with_production_features() -> None:
    """Every model feature must be computable for a competitor pair."""
    package = pytest.importorskip("joblib").load(
        "models/registry/matcher_ranked_v5_calibrated.joblib"
    )
    reranker = CompetitorReranker(package)
    assert set(MODEL_FEATURE_COLUMNS).issubset(set(reranker._feature_columns))

    prepared, master, _ = _frames()
    offer = prepared[prepared["offerid"].eq("comp-a")].iloc[0]
    master_rows = [row for _, row in master.iterrows()]
    ranked = reranker.rank(offer, master_rows)

    assert set(ranked) == {"SKU-CHICKEN-SAMOSA", "SKU-CHICKEN-NUGGETS"}
    assert sorted(entry.rank for entry in ranked.values()) == [1, 2]
    for entry in ranked.values():
        assert entry.raw_margin == entry.raw_margin  # finite, not NaN


def test_reranking_is_deterministic() -> None:
    """Same input, same order - twice."""
    package = pytest.importorskip("joblib").load(
        "models/registry/matcher_ranked_v5_calibrated.joblib"
    )
    reranker = CompetitorReranker(package)
    prepared, master, _ = _frames()
    offer = prepared[prepared["offerid"].eq("comp-a")].iloc[0]
    master_rows = [row for _, row in master.iterrows()]

    first = reranker.rank(offer, master_rows)
    second = reranker.rank(offer, master_rows)
    assert {k: v.rank for k, v in first.items()} == {
        k: v.rank for k, v in second.items()
    }
    assert {k: v.raw_margin for k, v in first.items()} == {
        k: v.raw_margin for k, v in second.items()
    }


# --------------------------------------------------------------------------
# C. Brand stripping, for arbitrary brands
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "brand", ["Alpha", "Beta", "Gamma Foods", "Zeta-Mart", "Al Kabeer"]
)
def test_brand_stripping_removes_any_brand_not_just_one(brand: str) -> None:
    text = f"{brand} Chicken Samosa 240gm"
    stripped = strip_competitor_brand(text, brand, protected="Samosa-Frozen")
    for token in brand.lower().replace("-", " ").split():
        assert token not in stripped.split()
    assert "chicken" in stripped and "samosa" in stripped


def test_brand_token_that_is_also_the_product_is_protected() -> None:
    """A brand word carrying product meaning must survive."""
    stripped = strip_competitor_brand(
        "Tempura Chicken Fries 400gm", "Tempura", protected="Tempura-Frozen"
    )
    assert "tempura" in stripped.split()


def test_transliterations_are_folded_for_any_brand() -> None:
    assert "samosa" in strip_competitor_brand(
        "Omega Frozen Sambosa 240gm", "Omega"
    ).split()


# --------------------------------------------------------------------------
# E. Hard conflicts survive the model
# --------------------------------------------------------------------------
def test_model_cannot_resurrect_a_hard_conflict() -> None:
    """Protein, form and pack conflicts are decided before ranking.

    The stub ranks every candidate it is given, so if ML could promote a row
    the conflicting offers would appear. They must not.
    """
    ruled = _run(reranker=None).long_format
    reranked = _run(reranker=_StubReranker()).long_format

    for frame in (ruled, reranked):
        statuses = frame.set_index("competitor_offer_id")[
            "competitor_match_status"
        ]
        assert statuses["comp-beef"] == "HARD_CONFLICT"
        assert statuses["comp-bigpack"] == "HARD_CONFLICT"
        # A different category is gated out before evaluation, so the nuggets
        # offer is absent rather than rejected. Absence is the stronger
        # guarantee: the model is never even offered the chance to promote it.
        if "comp-nugget" in statuses.index:
            assert statuses["comp-nugget"] != "MATCHED"

    def supported(frame: pd.DataFrame) -> set[str]:
        return set(
            frame.loc[
                frame["competitor_match_status"].isin(
                    SUPPORTED_COMPETITOR_STATUSES
                ),
                "competitor_offer_id",
            ]
        )

    assert supported(ruled) == supported(reranked)


def test_reranking_changes_order_without_changing_membership() -> None:
    """The whole point of a re-ranker, pinned as a property."""
    stub = _StubReranker()
    ruled = _run(reranker=None).long_format
    reranked = _run(reranker=stub).long_format
    assert stub.calls, "the re-ranker was never consulted"

    def rows(frame: pd.DataFrame) -> set[tuple[str, str]]:
        supported = frame[
            frame["competitor_match_status"].isin(
                SUPPORTED_COMPETITOR_STATUSES
            )
        ]
        return set(zip(supported["master_sku"], supported["competitor_offer_id"]))

    assert rows(ruled) == rows(reranked)
    assert (reranked["competitor_ranking_source"] == "lightgbm").any()


# --------------------------------------------------------------------------
# H. Thresholds must not leak
# --------------------------------------------------------------------------
def test_own_brand_thresholds_are_never_used_as_competitor_probabilities() -> None:
    """The re-ranker must not read the package's calibrated probability path."""
    source = (
        pytest.importorskip("pathlib")
        .Path("src/sku_mapping/competitors/reranker.py")
        .read_text(encoding="utf-8")
    )
    assert "predict_calibrated_proba" not in source.replace(
        "``predict_calibrated_proba``", ""
    )
    assert "auto_match_threshold" not in source
    assert "predict_raw_score" in source


def test_audit_records_score_semantics_not_a_probability() -> None:
    result = _run(reranker=_StubReranker())
    ml = result.diagnostics["ml_reranking"]
    assert ml["enabled"] is True
    assert ml["used_for"] == "ranking_only"
    assert ml["score_semantics"] == "raw_margin_within_offer_shortlist"


# --------------------------------------------------------------------------
# L. Fallback
# --------------------------------------------------------------------------
def test_missing_model_falls_back_to_rule_ordering() -> None:
    """An unavailable package degrades the run, it does not fail it."""
    assert (
        load_competitor_reranker(
            registry_path="does/not/exist.json",
            model_directory="does/not/exist",
            model_id="nope",
        )
        is None
    )
    result = _run(reranker=None)
    assert not result.long_format.empty
    assert result.diagnostics["ml_reranking"]["enabled"] is False
    assert (result.long_format["competitor_ranking_source"] == "rules").all()


def test_package_without_base_features_is_refused() -> None:
    with pytest.raises(CompetitorRerankerError):
        CompetitorReranker({"predictor": object(), "feature_columns": ["a"]})


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------
def test_audit_schema_carries_ml_columns_and_export_does_not() -> None:
    from sku_mapping.competitors.discovery import COMPETITOR_EXPORT_COLUMNS

    for column in (
        "competitor_lightgbm_score",
        "competitor_lightgbm_rank",
        "competitor_ranking_source",
    ):
        assert column in COMPETITOR_LONG_COLUMNS
        assert column not in COMPETITOR_EXPORT_COLUMNS


def test_batched_and_single_ranking_agree() -> None:
    """rank_many is an optimisation, so it must equal rank offer-by-offer."""
    package = pytest.importorskip("joblib").load(
        "models/registry/matcher_ranked_v5_calibrated.joblib"
    )
    reranker = CompetitorReranker(package)
    prepared, master, _ = _frames()
    master_rows = [row for _, row in master.iterrows()]
    offers = [
        (str(row["offerid"]), row, master_rows)
        for _, row in prepared.iterrows()
        if not bool(row.get("is_own"))
    ]
    batched = reranker.rank_many(offers)
    for offer_id, offer_row, rows in offers:
        one = reranker.rank(offer_row, rows)
        assert {k: v.rank for k, v in batched[offer_id].items()} == {
            k: v.rank for k, v in one.items()
        }
        for code, entry in one.items():
            assert batched[offer_id][code].raw_margin == pytest.approx(
                entry.raw_margin
            )


def test_configured_shortlist_k_overrides_the_package_default() -> None:
    """A documented knob that does nothing is worse than no knob."""
    import dataclasses

    from sku_mapping.competitors.discovery import _build_ml_context

    _, master, _ = _frames()
    base = load_config("config/default.yaml").competitors
    stub = _StubReranker()

    default_context = _build_ml_context(stub, master, base)
    assert default_context.top_k == stub.retrieval_k

    configured = dataclasses.replace(base, ml_shortlist_top_k=7)
    assert _build_ml_context(stub, master, configured).top_k == 7
