"""Rank competitor candidates with the existing own-brand LightGBM package.

The model is used as a RE-RANKER and nothing else. It reorders a shortlist the
rules have already produced; it never admits a candidate, never rejects one,
and its output is never read as a probability.

Why only ranking
----------------
The package is ``ranked-v5-cal``, trained on Al Kabeer offer -> Al Kabeer
master pairs. Three properties of that training make its *score* meaningless
for competitors while its *ordering* stays useful:

* Its thresholds (auto 0.99, review 0.94) were fitted for own-brand identity
  precision on an own-brand calibration split. No competitor pair was in it.
* Its isotonic calibrator emits ~16 distinct values over real data, so the
  calibrated number has almost no resolution for a task it was not fitted on.
* Training injected the true master into every candidate group. A competitor
  offer usually has no true Al Kabeer master at all - a case the model never
  saw - so it still reports high confidence for the best of a bad shortlist.

Ordering survives all three because it is relative within one offer's
shortlist, which is exactly what the group-relative features describe. So this
module reads ``predict_raw_score`` and deliberately never calls
``predict_calibrated_proba``.

Feature generation is the production own-brand code
(:func:`build_feature_vector`, :func:`build_extra_features`,
:func:`add_rank_features`) called through the same sequence the shadow
pipeline uses. Nothing is reimplemented here; the only competitor-specific
step is brand normalisation of the offer text, which is applied by rewriting
the offer row before featurisation rather than by forking the feature code.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sku_mapping.competitors.text_normalisation import strip_competitor_brand
from sku_mapping.constants import MODEL_FEATURE_COLUMNS
from sku_mapping.features.discriminative_features import build_extra_features
from sku_mapping.features.feature_generator import build_feature_vector
from sku_mapping.features.rank_features import add_rank_features

LOGGER = logging.getLogger(__name__)

#: Value the shadow pipeline substitutes for an unavailable feature. Repeated
#: here so a competitor feature frame is filled exactly as an own-brand one is.
MISSING_FEATURE_VALUE = -1.0


@dataclass(frozen=True)
class RerankedCandidate:
    """One master SKU's position in a single competitor offer's shortlist."""

    itemcode: str
    #: LightGBM raw margin (log-odds before any calibration). Comparable only
    #: WITHIN one offer's shortlist. Not a probability, not comparable across
    #: offers, and never to be thresholded - see the module docstring.
    raw_margin: float
    #: 1 = best candidate for this offer.
    rank: int


class CompetitorRerankerError(RuntimeError):
    """Raised when a re-ranker is asked for but cannot be built."""


class CompetitorReranker:
    """Order a competitor offer's master-SKU shortlist with the ML package."""

    def __init__(
        self,
        registered: Any,
        *,
        strip_brand: bool = True,
    ) -> None:
        package = getattr(registered, "package", registered)
        try:
            self._predictor = package["predictor"]
            self._feature_columns = list(package["feature_columns"])
        except (KeyError, TypeError) as error:  # pragma: no cover - defensive
            raise CompetitorRerankerError(
                "Model package is missing predictor or feature_columns"
            ) from error
        missing = [
            column
            for column in MODEL_FEATURE_COLUMNS
            if column not in self._feature_columns
        ]
        if missing:
            raise CompetitorRerankerError(
                "Model package does not expose the base feature columns: "
                + ", ".join(missing)
            )
        self._package = package
        self._strip_brand = bool(strip_brand)
        self.model_id = str(package.get("model_id", "unknown"))
        #: Group-relative columns are only meaningful when a whole shortlist is
        #: featurised together, which :meth:`rank` always does.
        self.requires_group_features = bool(
            package.get("requires_group_features", False)
        )
        self.retrieval_k = int(package.get("retrieval_k") or 0)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_registry(
        cls,
        *,
        registry_path: str | Path,
        model_directory: str | Path,
        model_id: str | None = None,
        package_reference: str | Path | None = None,
        require_package_status: str = "SHADOW_MODE_ONLY",
        strip_brand: bool = True,
    ) -> "CompetitorReranker":
        """Load through the production registry loader, not raw joblib.

        Going through the registry keeps the same provenance and status checks
        the own-brand path gets: an unregistered or wrong-status package is
        refused here exactly as it would be there.
        """
        from sku_mapping.shadow.predictor import load_registered_shadow_package

        registered = load_registered_shadow_package(
            registry_path=registry_path,
            model_directory=model_directory,
            model_id=model_id,
            package_reference=package_reference,
            require_package_status=require_package_status,
        )
        return cls(registered, strip_brand=strip_brand)

    # -- ranking ---------------------------------------------------------
    def _offer_for_features(self, offer_row: Mapping[str, Any]) -> dict[str, Any]:
        """Copy the offer with its own brand tokens removed from the name.

        Rewriting the row - rather than forking ``build_feature_vector`` - is
        what keeps competitor and own-brand featurisation on one code path.
        """
        row = dict(offer_row)
        if not self._strip_brand:
            return row
        brand = row.get("Brand Name", "")
        product = row.get("Product", "")
        row["Offer Name"] = strip_competitor_brand(
            row.get("Offer Name", ""), brand, protected=product
        )
        return row

    def rank_many(
        self,
        offers: Sequence[tuple[str, Mapping[str, Any], Sequence[Mapping[str, Any]]]],
    ) -> dict[str, dict[str, RerankedCandidate]]:
        """Rank several offers in one model call.

        Each offer keeps its own shortlist - the group-relative features are
        still computed per offer - but the feature frames are stacked so the
        predictor runs once instead of once per offer. Inference was never the
        bottleneck; the per-call overhead around it was.
        """
        blocks: list[pd.DataFrame] = []
        keys: list[tuple[str, list[str]]] = []
        for offer_id, offer_row, master_rows in offers:
            frame, itemcodes = self._feature_block(offer_row, master_rows)
            if frame is None:
                continue
            blocks.append(frame)
            keys.append((offer_id, itemcodes))
        if not blocks:
            return {}
        stacked = pd.concat(blocks, ignore_index=True)
        features = stacked.reindex(columns=self._feature_columns).astype(float)
        margins = np.asarray(
            self._predictor.predict_raw_score(
                features.fillna(MISSING_FEATURE_VALUE)
            ),
            dtype=float,
        )
        results: dict[str, dict[str, RerankedCandidate]] = {}
        cursor = 0
        for offer_id, itemcodes in keys:
            width = len(itemcodes)
            results[offer_id] = _to_ranked(
                itemcodes, margins[cursor : cursor + width]
            )
            cursor += width
        return results

    def _feature_block(
        self,
        offer_row: Mapping[str, Any],
        master_rows: Sequence[Mapping[str, Any]],
    ) -> tuple[pd.DataFrame | None, list[str]]:
        """Featurise one offer's whole shortlist, group features included."""
        if not master_rows:
            return None, []
        featurised = self._offer_for_features(offer_row)
        rows: list[dict[str, float | int | None]] = []
        itemcodes: list[str] = []
        for master_row in master_rows:
            itemcode = str(master_row.get("Itemcode", "")).strip()
            if not itemcode:
                continue
            try:
                base = build_feature_vector(featurised, master_row)
            except Exception:  # pragma: no cover - defensive, mirrors pipeline
                LOGGER.exception(
                    "Competitor feature generation failed; candidate skipped"
                )
                continue
            row: dict[str, float | int | None] = {
                column: base[column] for column in MODEL_FEATURE_COLUMNS
            }
            if self.requires_group_features:
                row.update(
                    build_extra_features(
                        _pair_text(
                            featurised.get("Offer Name"),
                            featurised.get("Product"),
                        ),
                        _pair_text(
                            master_row.get("Itemname"),
                            master_row.get("Item-Spec"),
                        ),
                    )
                )
            rows.append(row)
            itemcodes.append(itemcode)
        if not rows:
            return None, []
        frame = pd.DataFrame(rows).astype(float).fillna(MISSING_FEATURE_VALUE)
        if self.requires_group_features:
            add_rank_features(frame)
        return frame, itemcodes

    def rank(
        self,
        offer_row: Mapping[str, Any],
        master_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, RerankedCandidate]:
        """Rank one offer's shortlist, best first.

        ``master_rows`` must be the WHOLE shortlist for this one offer. Passing
        a partial list, or rows belonging to several offers, silently changes
        the group-relative features and therefore the ordering.

        Returns an empty mapping when there is nothing to rank, so a caller can
        treat "no ML opinion" and "ML unavailable" identically.
        """
        frame, itemcodes = self._feature_block(offer_row, master_rows)
        if frame is None:
            return {}
        features = frame.reindex(columns=self._feature_columns).astype(float)
        margins = np.asarray(
            self._predictor.predict_raw_score(
                features.fillna(MISSING_FEATURE_VALUE)
            ),
            dtype=float,
        )
        if margins.shape != (len(itemcodes),):  # pragma: no cover - defensive
            raise CompetitorRerankerError("Re-ranker returned invalid shapes")
        return _to_ranked(itemcodes, margins)


def _to_ranked(
    itemcodes: Sequence[str], margins: "np.ndarray"
) -> dict[str, RerankedCandidate]:
    """Order one offer's shortlist, ties broken by itemcode for determinism."""
    order = sorted(
        range(len(itemcodes)),
        key=lambda index: (-float(margins[index]), itemcodes[index]),
    )
    return {
        itemcodes[index]: RerankedCandidate(
            itemcode=itemcodes[index],
            raw_margin=float(margins[index]),
            rank=position,
        )
        for position, index in enumerate(order, start=1)
    }


def _pair_text(*parts: object) -> str:
    return " ".join(str(part) for part in parts if part not in (None, ""))


def load_competitor_reranker(
    *,
    registry_path: str | Path,
    model_directory: str | Path,
    model_id: str | None = None,
    package_reference: str | Path | None = None,
    require_package_status: str = "SHADOW_MODE_ONLY",
    strip_brand: bool = True,
) -> CompetitorReranker | None:
    """Build a re-ranker, or return None when one cannot be had.

    Competitor discovery is a rules pipeline that ML only reorders, so a
    missing or unloadable model is a degraded run rather than a failed one:
    the caller falls back to the rules' own ordering and records that it did.
    """
    try:
        return CompetitorReranker.from_registry(
            registry_path=registry_path,
            model_directory=model_directory,
            model_id=model_id,
            package_reference=package_reference,
            require_package_status=require_package_status,
            strip_brand=strip_brand,
        )
    except Exception:
        LOGGER.warning(
            "Competitor ML re-ranking unavailable; falling back to rule order",
            exc_info=True,
        )
        return None
