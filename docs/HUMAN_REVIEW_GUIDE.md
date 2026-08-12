# Shadow Human-Review Guide

Shadow review files are offer-centred and observational. Completing a review
never changes a production match and never adds data to model training.

## Allowed labels

- `CORRECT_TOP_CANDIDATE`: the first listed candidate is correct. The selected
  item code must equal `top_candidate_1_itemcode`.
- `CORRECT_OTHER_CANDIDATE`: another Product Master SKU is correct. The
  dashboard permits selecting any current Product Master item, including one
  outside the supplied top-K list.
- `NO_VALID_MASTER_SKU`: none of the listed candidates or Product Master items
  is a valid match. Leave the selected item code blank.
- `AMBIGUOUS_OFFER`: the offer supports no defensible single interpretation.
  Leave the selected item code blank.
- `MULTIPLE_VALID_SKUS`: more than one SKU is legitimately valid. Leave the
  single selected item code blank and explain the alternatives in notes.
- `INSUFFICIENT_INFORMATION`: the offer lacks enough evidence. Leave the
  selected item code blank.
- `DATA_QUALITY_ERROR`: source or catalogue data is materially incorrect.
  Leave the selected item code blank and document the issue.

Every completed row requires `reviewer_code` and an ISO-8601
`review_timestamp`. Intake rejects invalid labels, contradictory selections,
unknown Product Master codes, and duplicate reviews. Submitted files and
normalised review records are immutable. Accepted records remain in staging
until a separately approved challenge-set process uses them.

## Assisted-mode review

Assisted mode is an explicit deployment setting, not a claim that the model
has an approved production threshold. A candidate with calibrated probability
below the configured auto-accept threshold is sent to `MANUAL_REVIEW`; it is
not converted to `NO_MATCH`. A high probability is also sent to review when a
protein, mixed-protein, family, known size/weight, pack-format, feature,
catalogue, package, or prediction conflict blocks automatic acceptance.

Every assisted audit record identifies the run, offer, candidate, model
package hash, raw and calibrated probabilities, configured threshold,
`threshold_source=user_configured`,
`production_threshold_approved=false`, final decision, reason, override, and
conflict flags. Review completion still follows the labels and immutable
intake process above. Reviews are never learned online and do not enter
training without a separate governed training phase.

For decomposed multi-product offers, each entity is reviewed independently.
The screen retains the complete original offer, entity index/count,
conjunction, inherited-attribute flags and parser confidence. Reviewers may
confirm the decomposition, request a merge or further split, correct entity
text/attributes, or mark it genuinely mixed/ambiguous. These corrections and
the reviewer-selected SKU are stored separately from the immutable original
machine proposal.

## Embedding second-opinion fields

When embedding scoring is explicitly enabled for a shadow or assisted
monitoring run, each retained candidate also includes the embedding model ID
and version, exact prepared offer and candidate text, cosine similarity,
within-offer embedding rank, top-candidate flag, and an explicit failure
reason when unavailable. Embedding retrieval may add a bounded candidate to
the fuzzy set, but it does not change the frozen LightGBM feature schema,
safety overrides, human-review label, or training intake behavior.

## Agreement routing fields

Phase 6C compares the LightGBM and embedding rankings after both scorers have
processed the same retained candidate set. The offer-level agreement record
includes each scorer's top SKU, score, rank and score margin, explicit
conflict flags, an agreement status, a routing decision, reason codes, and a
human-readable routing reason.

`AUTO_ACCEPT` additionally requires the separate
`agreement.allow_embedding_auto_accept` safety approval. It is disabled by
default because the audited lexical encoder is approved for retrieval/ranking,
not auto-match authority. Even when enabled, both outputs must be valid, both
scorers must select the same existing master SKU, calibrated LightGBM must
meet the configured threshold, and no hard conflict may be present. Agreement
is supporting evidence, not proof of correctness. Disagreement and weak agreement are routed
to `LLM_REVIEW`; Phase 6C records that route but does not call an LLM. Hard
conflicts route to `MANUAL_REVIEW`. An unavailable scorer produces an explicit
unavailable status and safe fallback rather than false agreement.

Agreement routes are observational in this phase. They do not alter the
authoritative production choice, completed-review intake, or learning dataset.

## Structured LLM review fields

When Phase 6D is explicitly enabled, only offers routed to `LLM_REVIEW` are
sent to the configured bounded provider. The review artifact records status,
parsed decision, supplied selected SKU, confidence, controlled reason codes,
validation errors, latency, retries, cache use, deterministic conflict
enforcement, and the final diagnostic route.

`LLM_ACCEPT` is eligibility evidence only and does not modify the production
mapping. Low confidence, uncertainty, invalid output, timeout, provider
failure, and deterministic hard conflicts route to manual review. Reviewers
must continue using the governed human labels above; LLM output is not a human
label and never enters training automatically.

## Unified assisted decisions

Phase 6E may apply `AUTO_ACCEPT` or `LLM_ACCEPT` only when the corresponding
deterministic policy is satisfied. Both decisions retain full scorer and
review provenance. `MANUAL_REVIEW`, `NO_CANDIDATE`, `MODEL_ERROR`, invalid
LLM output, and hard-conflict rows remain ineligible mappings and cannot be
used for competitor discovery.

An `LLM_ACCEPT` is still not a human label. Human-review status and completed
review intake remain separate, and neither inference nor upload processing
triggers model fitting or training-data updates.
