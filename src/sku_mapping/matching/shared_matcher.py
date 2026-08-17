"""One matching engine for both offer populations.

An Al Kabeer offer and a rival's offer ask the same question of the same
catalogue: which Product_Master.xlsx SKU is this product? The only difference
is which side of ``is_own`` the offer sits on, so there is one candidate
generator, one 41-feature block, one model, and one threshold here rather than
a second pipeline for competitors.

What the populations mean differs, and that difference is worth stating:

* OWN - "this offer IS this Al Kabeer SKU". The mapping is an identity.
* COMPETITOR - "this rival offer is the same product as this Al Kabeer SKU",
  which is precisely the competitive relationship. The rival product is not in
  the master catalogue and never will be; a high score means the two products
  are interchangeable to a shopper.

The features carry no notion of who sells the product. All 41 compare an offer
against a master row - semantics, text similarity, measurements, offer
composition, and within-shortlist ranks - and not one reads a brand, retailer,
price, or ``is_own``. That is why the same model can answer both.

The threshold is the existing own-brand 0.95. It is NOT re-derived here. A
real-data check on 400 offers of each population found top-1 calibrated scores
at or above 0.95 for 29.0% of own-brand offers and 24.5% of competitor offers,
with matching distribution shapes, so the cut behaves comparably on both. That
is a distribution check, not an accuracy claim: no human competitor labels
exist, so competitor precision at 0.95 is unmeasured.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from sku_mapping.matching.routing import RouteOutcome, RoutingMode

LOGGER = logging.getLogger(__name__)

#: The own-brand production threshold, reused unchanged. Deliberately not a new
#: competitor-specific number.
SHARED_AUTO_ACCEPT_THRESHOLD = 0.95


class OfferPopulation(str, Enum):
    """Which side of ``is_own`` an offer sits on."""

    OWN = "own"
    COMPETITOR = "competitor"

    @property
    def is_own_value(self) -> bool:
        return self is OfferPopulation.OWN


class MatchDecisionOutcome(str, Enum):
    """Terminal outcome for one offer. No production human route."""

    AUTO_ACCEPT = "AUTO_ACCEPT"
    LLM_ACCEPT = "LLM_ACCEPT"
    REJECTED = "REJECTED"
    NO_CANDIDATE = "NO_CANDIDATE"
    #: Only reachable with the global LLM toggle OFF. An explicit manual queue,
    #: not a failure - and never produced when the toggle is ON.
    HUMAN_VALIDATION = "HUMAN_VALIDATION"


@dataclass(frozen=True)
class ScoredCandidate:
    """One master SKU scored for one offer."""

    itemcode: str
    master_name: str
    calibrated_score: float
    rank: int


@dataclass(frozen=True)
class OfferMatch:
    """The decision for one offer, with the shortlist that produced it."""

    offer_id: str
    offer_name: str
    population: OfferPopulation
    outcome: MatchDecisionOutcome
    matched_master_sku: str | None
    score: float | None
    reason: str
    candidates: tuple[ScoredCandidate, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.outcome in {
            MatchDecisionOutcome.AUTO_ACCEPT,
            MatchDecisionOutcome.LLM_ACCEPT,
        }


@dataclass
class MatchRunStats:
    """Counters a run can be judged on without re-reading the frame."""

    offers_processed: int = 0
    candidate_rows: int = 0
    ml_evaluations: int = 0
    at_or_above_threshold: int = 0
    below_threshold: int = 0
    no_candidate: int = 0
    human_validation: int = 0
    llm_calls: int = 0
    llm_accepts: int = 0
    llm_rejects: int = 0
    llm_failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class MatchResult:
    """Everything one population's run produced."""

    population: OfferPopulation
    matches: tuple[OfferMatch, ...]
    stats: MatchRunStats
    threshold: float = SHARED_AUTO_ACCEPT_THRESHOLD
    #: The mode this run used. Recorded so an output can always be traced back
    #: to the toggle state that produced it.
    mode: "Any | None" = None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "offer_id": match.offer_id,
                    "offer_name": match.offer_name,
                    "population": match.population.value,
                    "outcome": match.outcome.value,
                    "matched_master_sku": match.matched_master_sku,
                    "score": match.score,
                    "reason": match.reason,
                    "shortlist_size": len(match.candidates),
                }
                for match in self.matches
            ],
            columns=[
                "offer_id",
                "offer_name",
                "population",
                "outcome",
                "matched_master_sku",
                "score",
                "reason",
                "shortlist_size",
            ],
        )

    @property
    def accepted(self) -> tuple[OfferMatch, ...]:
        return tuple(match for match in self.matches if match.accepted)


def select_population(
    offers: pd.DataFrame, population: OfferPopulation
) -> pd.DataFrame:
    """Split the dump on ``is_own`` and nothing else.

    No brand string is matched here. ``is_own`` already carries the
    classification, and re-deriving it from names would let the two disagree.
    A missing value is treated as not-own, which keeps an unclassified offer
    out of the Al Kabeer population rather than smuggling it in.
    """
    if "is_own" not in offers.columns:
        raise KeyError(
            "Offers must carry is_own to be split into populations; "
            "preprocess_clickflyer produces it"
        )
    flag = offers["is_own"].fillna(False).astype(bool)
    return offers[flag if population.is_own_value else ~flag]


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


class SharedMatcher:
    """Candidate generation, 41 features, model, and threshold - once.

    The featuriser is the competitor re-ranker's block builder, which already
    assembles the base, discriminative, and group-relative columns for a whole
    shortlist. It is reused rather than re-implemented so the two populations
    cannot drift apart in how they are featurised.
    """

    def __init__(
        self,
        package: Mapping[str, Any],
        product_master: pd.DataFrame,
        *,
        threshold: float = SHARED_AUTO_ACCEPT_THRESHOLD,
        top_k: int | None = None,
    ) -> None:
        from sku_mapping.competitors.reranker import CompetitorReranker
        from sku_mapping.matching.candidate_generator import CandidateGenerator

        self._package = package
        self._predictor = package["predictor"]
        self._feature_columns = list(package["feature_columns"])
        self.threshold = float(threshold)
        self.model_id = str(package.get("model_id", "unknown"))
        master = product_master.reset_index(drop=True)
        self._generator = CandidateGenerator(master)
        self._master_lookup = {
            _text(row["Itemcode"]): row for _, row in master.iterrows()
        }
        self.top_k = int(
            top_k or package.get("retrieval_k") or 20
        )
        # Brand stripping is population-dependent: a rival's brand can never
        # appear in Al Kabeer master text, so leaving it in penalises them for
        # naming themselves. An Al Kabeer offer's brand legitimately matches
        # the catalogue, so stripping it there would destroy real evidence.
        self._featurisers = {
            OfferPopulation.OWN: CompetitorReranker(package, strip_brand=False),
            OfferPopulation.COMPETITOR: CompetitorReranker(
                package, strip_brand=True
            ),
        }

    def score_offers(
        self, offers: pd.DataFrame, population: OfferPopulation
    ) -> tuple[dict[str, tuple[ScoredCandidate, ...]], int]:
        """Return each offer's scored shortlist, best first, plus ML row count."""
        from sku_mapping.competitors.reranker import MISSING_FEATURE_VALUE

        if offers.empty:
            return {}, 0
        featuriser = self._featurisers[population]
        shortlists = self._generator.generate_candidates_batch(
            offers, top_k=self.top_k
        )
        blocks: list[pd.DataFrame] = []
        keys: list[tuple[str, list[str]]] = []
        for (_, offer), candidates in zip(offers.iterrows(), shortlists):
            master_rows = [
                self._master_lookup[candidate.itemcode]
                for candidate in candidates
                if candidate.itemcode in self._master_lookup
            ]
            if not master_rows:
                continue
            frame, itemcodes = featuriser._feature_block(offer, master_rows)
            if frame is None:
                continue
            blocks.append(frame)
            keys.append((_text(offer.get("offerid")), itemcodes))
        if not blocks:
            return {}, 0
        stacked = pd.concat(blocks, ignore_index=True)
        features = (
            stacked.reindex(columns=self._feature_columns)
            .astype(float)
            .fillna(MISSING_FEATURE_VALUE)
        )
        scores = np.asarray(
            self._predictor.predict_calibrated_proba(features), dtype=float
        )
        results: dict[str, tuple[ScoredCandidate, ...]] = {}
        cursor = 0
        for offer_id, itemcodes in keys:
            width = len(itemcodes)
            window = scores[cursor : cursor + width]
            cursor += width
            # Ties break on itemcode so two runs cannot disagree about which of
            # two identically scored masters came first.
            ordered = sorted(
                zip(itemcodes, window),
                key=lambda pair: (-float(pair[1]), pair[0]),
            )
            results[offer_id] = tuple(
                ScoredCandidate(
                    itemcode=itemcode,
                    master_name=_text(
                        self._master_lookup.get(itemcode, {}).get("Itemname")
                        if itemcode in self._master_lookup
                        else ""
                    ),
                    calibrated_score=float(score),
                    rank=position,
                )
                for position, (itemcode, score) in enumerate(ordered, start=1)
            )
        return results, len(features)


def match_offers(
    offers: pd.DataFrame,
    product_master: pd.DataFrame,
    *,
    package: Mapping[str, Any],
    population: OfferPopulation | str = OfferPopulation.OWN,
    mode: "RoutingMode | None" = None,
    adjudicator: Any | None = None,
    top_k: int | None = None,
    matcher: "SharedMatcher | None" = None,
) -> MatchResult:
    """Match one population of offers against the master catalogue.

    The only population-dependent inputs are the ``is_own`` filter and whether
    the offer's own brand is stripped before featurisation. Everything after
    that - retrieval, features, model, threshold, routing - is shared.

    Offers at or above the threshold are accepted automatically. Everything
    below goes to the adjudicator, and anything the adjudicator cannot settle
    is REJECTED: there is no production human route.
    """
    from sku_mapping.competitors.adjudicator import (
        AdjudicationCandidate,
        AdjudicationRequest,
    )
    from sku_mapping.competitors.policy import CompetitorDecisionReason

    population = OfferPopulation(population)
    # One canonical source for the threshold and the review destination. The
    # matcher never picks a number itself.
    mode = mode or RoutingMode.from_toggle(adjudicator is not None)
    threshold = mode.auto_accept_threshold
    stats = MatchRunStats()
    selected = select_population(offers, population)
    if "offerid" in selected.columns:
        selected = selected.drop_duplicates("offerid")
    stats.offers_processed = int(len(selected))
    if selected.empty:
        return MatchResult(population, (), stats, threshold, mode)

    engine = matcher or SharedMatcher(
        package, product_master, threshold=threshold, top_k=top_k
    )
    shortlists, ml_rows = engine.score_offers(selected, population)
    stats.candidate_rows = ml_rows
    stats.ml_evaluations = ml_rows

    matches: list[OfferMatch] = []
    for _, offer in selected.iterrows():
        offer_id = _text(offer.get("offerid"))
        offer_name = _text(offer.get("Offer Name"))
        candidates = shortlists.get(offer_id, ())
        if not candidates:
            stats.no_candidate += 1
            matches.append(
                OfferMatch(
                    offer_id, offer_name, population,
                    MatchDecisionOutcome.NO_CANDIDATE, None, None,
                    "NO_CANDIDATE_RETRIEVED",
                )
            )
            continue

        best = candidates[0]
        if mode.decide(best.calibrated_score) is RouteOutcome.AUTO_ACCEPT:
            stats.at_or_above_threshold += 1
            matches.append(
                OfferMatch(
                    offer_id, offer_name, population,
                    MatchDecisionOutcome.AUTO_ACCEPT, best.itemcode,
                    best.calibrated_score, "AT_OR_ABOVE_THRESHOLD", candidates,
                )
            )
            continue

        stats.below_threshold += 1
        # Toggle OFF: this is where the run stops being automatic. No provider
        # is consulted and none is even constructed, so an OFF run needs no API
        # key and makes no call.
        if not mode.routes_to_llm:
            stats.human_validation += 1
            matches.append(
                OfferMatch(
                    offer_id, offer_name, population,
                    MatchDecisionOutcome.HUMAN_VALIDATION, None,
                    best.calibrated_score, "BELOW_THRESHOLD_HUMAN_VALIDATION",
                    candidates,
                )
            )
            continue
        if adjudicator is None:
            matches.append(
                OfferMatch(
                    offer_id, offer_name, population,
                    MatchDecisionOutcome.REJECTED, None, best.calibrated_score,
                    CompetitorDecisionReason.LLM_DISABLED.value, candidates,
                )
            )
            continue

        verdict = adjudicator.adjudicate(
            AdjudicationRequest(
                offer_id=offer_id,
                offer_name=offer_name,
                competitor_brand=_text(offer.get("Brand Name")),
                competitor_product=_text(offer.get("Product")),
                competitor_variant=_text(offer.get("Variant")),
                competitor_pack_size=_text(offer.get("Base Packsize")),
                competitor_retailer=_text(offer.get("Retailer Name")),
                candidates=tuple(
                    AdjudicationCandidate(
                        master_sku=candidate.itemcode,
                        master_name=candidate.master_name,
                        model_score=candidate.calibrated_score,
                        model_rank=candidate.rank,
                    )
                    for candidate in candidates[
                        : getattr(adjudicator, "max_candidates", 5)
                    ]
                ),
                ambiguity_reason="BELOW_THRESHOLD",
                top_score=best.calibrated_score,
                runner_up_score=(
                    candidates[1].calibrated_score
                    if len(candidates) > 1
                    else None
                ),
            )
        )
        stats.llm_calls += 1
        if verdict.reason is CompetitorDecisionReason.LLM_ACCEPTED:
            stats.llm_accepts += 1
            matches.append(
                OfferMatch(
                    offer_id, offer_name, population,
                    MatchDecisionOutcome.LLM_ACCEPT, verdict.selected_master,
                    best.calibrated_score, verdict.reason.value, candidates,
                )
            )
            continue
        if verdict.reason in {
            CompetitorDecisionReason.LLM_REJECTED,
            CompetitorDecisionReason.LLM_UNCERTAIN,
        }:
            stats.llm_rejects += 1
        else:
            stats.llm_failures += 1
        matches.append(
            OfferMatch(
                offer_id, offer_name, population,
                MatchDecisionOutcome.REJECTED, None, best.calibrated_score,
                verdict.reason.value, candidates,
            )
        )
    return MatchResult(population, tuple(matches), stats, threshold, mode)


def competitors_for_master(
    competitor_result: MatchResult,
) -> dict[str, tuple[OfferMatch, ...]]:
    """Invert accepted competitor matches into per-master-SKU competitor lists.

    This is the whole of competitor discovery under the shared architecture:
    once every rival offer carries the master SKU it maps to, a target's
    competitors are just the offers that landed on it. No second engine, no
    target-by-offer sweep.
    """
    if competitor_result.population is not OfferPopulation.COMPETITOR:
        raise ValueError(
            "competitors_for_master expects a COMPETITOR population result; "
            f"got {competitor_result.population.value}"
        )
    grouped: dict[str, list[OfferMatch]] = {}
    for match in competitor_result.accepted:
        if match.matched_master_sku:
            grouped.setdefault(match.matched_master_sku, []).append(match)
    return {
        master_sku: tuple(
            sorted(items, key=lambda match: (-(match.score or 0.0), match.offer_id))
        )
        for master_sku, items in sorted(grouped.items())
    }
