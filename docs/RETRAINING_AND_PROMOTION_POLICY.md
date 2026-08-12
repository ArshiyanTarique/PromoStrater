# Retraining and Promotion Policy

## Scope

Phase 7C provides an explicit offline learning workflow:

```text
persisted predictions and reviews
  -> immutable labelled snapshot
  -> inert challenger package
  -> same-row champion/challenger comparison
  -> assisted-use registration only when every policy check passes
  -> separate explicit activation
```

This is not continuous learning. Upload processing, inference, the Streamlit
dashboard, and five-question review completion never call the trainer. A
training or comparison failure cannot change the registry activation pointer.

Automatic production matching remains disabled. A policy-passing challenger
is eligible only for assisted use and still requires a separate activation
command.

## Evidence gate

The default retraining gate is 50 new GOLD answers since the most recent
explicit model activation. Five post-upload questions do not trigger
retraining.

A lower development minimum requires both:

- `--minimum-gold-override`;
- a non-empty `--override-reason`.

The override is persisted in the snapshot manifest and learning database.
Production-like jobs should not use it.

## Training-label policy

Every snapshot contains the original trusted baseline feature table and the
eligible reviewed additions that remain after evaluation holdout and
connected-component exclusion.

| Trust source | Included by default | Weight |
|---|---:|---:|
| Baseline trusted rows | Yes | Existing provenance weight × 1.0 |
| GOLD human review | Yes | 1.0 |
| SILVER policy-qualified LLM acceptance | No | 0.25 when explicitly enabled |
| PSEUDO model/agreement output | No | 0.0 |
| REJECTED/inconclusive output | No | 0.0 |

Trust columns, source identifiers, raw offer text, and label provenance are
audit fields. LightGBM receives exactly `MODEL_FEATURE_COLUMNS`, in their
declared order.

GOLD answer materialization is explicit:

- True: the suggested candidate is positive.
- False with a supplied correction: the suggestion is negative and the
  corrected supplied candidate is positive.
- False with none-of-candidates: all supplied candidates are negative.
- Cannot-determine: excluded.

SILVER is restricted to structured LLM acceptance already marked
`POLICY_QUALIFIED_REVIEW_REQUIRED_BEFORE_TRAINING`. Enabling SILVER is an
explicit job-level choice and is recorded.

## Immutable snapshots

Snapshots are stored under:

```text
data/learning/training_snapshots/<dataset-id>/
```

Each directory contains `training_snapshot.parquet`,
`recent_gold_evaluation.parquet`, and `snapshot_manifest.json`.

The manifest records the dataset ID, UTC timestamp, baseline and artifact
hashes, canonical content hash, counts by trust level, included/excluded
review IDs, SILVER label IDs, feature schema, weights, override evidence,
leakage policy, and challenge-set exclusion proof.

Dataset IDs derive from canonical content. Existing artifacts are verified
and never overwritten.

The newest configured GOLD reviews are reserved as an evaluation holdout.
Every baseline or reviewed row in a connected leakage component touching that
holdout is removed from challenger training. The trainer then independently
creates leakage-safe train, validation, and calibration splits.

## Sealed challenge protection

Ordinary snapshot, training, and comparison paths use the existing
sealed-challenge guard. A path inside a `SEALED_UNOPENED` challenge directory
is rejected before Parquet is loaded.

Snapshot creation may read only challenge manifests to produce exclusion
proofs. It never reads sealed data artifacts. Included or held-out review
identities overlapping a sealed manifest abort the job.

Phase 7C does not open the sealed challenge set.

## Challenger fitting and calibration

Challenger fitting:

- uses `lightgbm.LGBMClassifier`;
- uses balanced class handling and explicit sample weights;
- selects hyperparameters and early-stops on validation only;
- fits calibration on the separate calibration split;
- tunes diagnostic thresholds on calibration only;
- does not inspect the recent GOLD evaluation holdout;
- saves a new immutable package under `models/challengers/<model-id>/`;
- records `CHALLENGER_TRAINED` in SQLite;
- does not register or activate the package.

## Champion–challenger comparison

Both models score exactly the same evaluation rows. The reserved recent GOLD
holdout is mandatory. Additional explicitly supplied unsealed evaluation
tables may be added.

The report includes precision, recall, F1, ROC-AUC, PR-AUC, Brier score, log
loss, expected calibration error, precision and coverage at 0.85, critical
protein/family/size/pack errors, product-family subgroups, recent human-label
performance, and row-level regression cases.

Aggregate improvement is insufficient. Every configured check must pass:

1. minimum evaluation evidence;
2. precision at 0.85 is not materially worse;
3. critical-error count is not worse;
4. Brier score is not materially worse;
5. expected calibration error is not materially worse;
6. no supported product-family subgroup exceeds regression tolerance;
7. coverage remains acceptable;
8. package validation passes;
9. required regression tests pass.

Skipped regression tests force rejection.

## Registration, activation, and rollback

A failed challenger is marked `REJECTED`; its package and reports remain
available. It is not copied to the main registry.

A passing challenger is copied immutably into `models/registry`, recorded as
`APPROVED_FOR_ASSISTED_USE`, and remains inactive. Automatic production
matching remains prohibited.

Activation is a separate compare-and-swap operation requiring the approved
model ID, expected current champion, actor, and reason. The registry stores
`active_assisted_model_id` plus append-only activation history. A stale
expected champion aborts activation. Rollback explicitly restores the
immediately previous champion.

Activation does not rewrite `config/default.yaml`, model packages, or source
data. Runtime deployment owners must deliberately consume the approved
pointer in their controlled environment.

## CLI

Build a snapshot:

```powershell
.\.venv\Scripts\python.exe -m sku_mapping.retraining `
  build-training-snapshot `
  --config config/default.yaml `
  --baseline data/processed/training_features.parquet
```

Development-only override:

```powershell
.\.venv\Scripts\python.exe -m sku_mapping.retraining `
  build-training-snapshot `
  --minimum-gold-override 10 `
  --override-reason "bounded local integration test"
```

Train an inert challenger:

```powershell
.\.venv\Scripts\python.exe -m sku_mapping.retraining `
  train-challenger `
  --snapshot-manifest <snapshot_manifest.json> `
  --champion-model-id <registered-champion-id>
```

Compare and conditionally register:

```powershell
.\.venv\Scripts\python.exe -m sku_mapping.retraining `
  compare-models `
  --champion-model-id <registered-champion-id> `
  --challenger-package <challenger.joblib> `
  --challenger-metadata <challenger.json> `
  --snapshot-manifest <snapshot_manifest.json>
```

Activate:

```powershell
.\.venv\Scripts\python.exe -m sku_mapping.retraining `
  activate-model `
  --model-id <approved-challenger-id> `
  --expected-current-model-id <champion-id> `
  --actor <operator-id> `
  --reason "approved controlled assisted rollout"
```

Rollback:

```powershell
.\.venv\Scripts\python.exe -m sku_mapping.retraining `
  activate-model --rollback `
  --actor <operator-id> `
  --reason "rollback after monitored regression"
```

## Limitations

- The current learning store has no real GOLD labels, so real retraining is
  not eligible.
- An unsealed recent human holdout is development evidence, not a substitute
  for an independently authorized sealed challenge evaluation.
- Product-family metadata is absent from some historic predictions; missing
  families are reported as `<unknown>`.
- SQLite and the JSON registry need single-host operational controls and
  backups.
- No Phase 7C artifact alone establishes full production readiness.
