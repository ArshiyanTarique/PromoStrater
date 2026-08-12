# Commercial Matching Correctness Design

## Scope and invariants

This design corrects inference evidence without redesigning the business-output
pipeline. RapidFuzz remains the candidate generator, LightGBM remains the
candidate scorer, and competitor discovery remains downstream of a confirmed
own-SKU match.

The deployed LightGBM contract is immutable:

- `MODEL_FEATURE_COLUMNS` remains the ordered list of 19 trained features.
- the saved model package and thresholds are not changed;
- no pre-model auto-accept path is introduced;
- no reviewed SKU, fixture row, or item code is encoded in production rules.

## New evidence boundary

`features/commercial_attributes.py` parses source and master rows independently
into the same structured representation. A source parser receives no master
row, so candidate attributes cannot leak into source evidence.

The representation records family/subfamily, protein, variants, flavour,
spiciness, product line, commercial format, base/bonus/total/unit measures,
pack and piece counts, bundle structure, ambiguity, source-field
contradictions, confidence, and provenance.

Measurement comparison is role-aware. Stable categories distinguish exact,
conversion-equivalent, tolerance, promotion, unit-weight, total-weight,
pack-format and unit-size states instead of accepting any numeric overlap.

## Outcome and safety policy

Each candidate receives `EXACT_MATCH`, `ADAPTED_MATCH`,
`UNACCEPTABLE_MATCH`, or `UNKNOWN`. The outcome is deterministic safety
evidence, not another probability model. Hard commercial contradictions feed
the existing safety gate. Adapted and ambiguous evidence remain review-only.
Existing final decisions and output fields remain backward compatible; new
columns are additive.

## Failure-class mapping

| Observed class | Generalized correction |
|---|---|
| Multi-product/slash ambiguity | explicit bundle and ambiguity states |
| Promotional structure mismatch | base/bonus/total and pack roles |
| Variant/flavour/spicy contradiction | polarity-aware comparison |
| Product-family/protein conflict | independent source/master taxonomy |
| Commercial format/unit construction | role-aware measures |
| Review evidence ambiguity | outcome plus stable reason codes |

## Risks and controls

- Incomplete taxonomy yields unknown evidence, never exact justification.
- Parsing adds work per retained candidate; source/master parses are cached.
- The historical 19 model features retain their legacy semantics until a
  separately governed retraining phase.
- Review corrections preserve both original proposal and replacement.

