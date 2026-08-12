# Learning Data Governance

## Scope

Phase 7A provides persistent observation and review storage. It does not train,
retrain, calibrate, register, activate, retire, or deploy a model. Upload-time
processing may write observations and create a review session, but it never
calls a trainer or changes model weights, Product Master data, source uploads,
or sealed challenge data.

Phase 7C adds a separate explicit offline consumer of this store. It does not
change upload-time behavior: inference and five-question completion still
never retrain. Snapshot creation requires the configured GOLD-label minimum,
and challenger training, comparison, registration, activation, and rollback
are separate audited commands. See `docs/RETRAINING_AND_PROMOTION_POLICY.md`.

The configured repository-owned database is
`data/learning/sku_learning.db`. SQLite sidecars and CSV exports under
`data/learning/` are runtime data and are excluded from source control.
Changing `learning_store.database_path` selects a different store.

## Stored provenance

The schema is versioned with ordered transactional migrations and SQLite
`PRAGMA user_version`. It stores:

- `pipeline_runs`: upload identity/hash, counts, deployment mode, model
  identities, threshold, stage runtimes, output paths, status, and errors.
- `predictions`: every retained offer/candidate pair, ranks, LightGBM and
  embedding scores, agreement, LLM result, final decision, conflict flags, and
  an exact shared-feature snapshot.
- `review_sessions`: one deterministic five-question session per successful
  run.
- `human_reviews`: question selection provenance, presentation/answer
  timestamps, reviewer response, correction, and label quality.
- `automated_labels`: model- or LLM-proposed labels and explicit eligibility.
- `model_versions`: observed immutable model identities and lifecycle
  metadata. Observation does not activate a model.
- `training_datasets`: prospective dataset manifests, review inclusion and
  exclusion IDs, content hash, label counts, and challenge-exclusion proof.

No secrets are stored. JSON fields use canonical, sorted serialization for
stable hashing and transparent export.

## Five-question review policy

After a successful run, a session is created only when at least five unique
offers have retained candidates. Exactly five distinct offers are selected:

1. a high-confidence `AUTO_ACCEPT`;
2. a result nearest the configured 0.85 policy threshold;
3. a LightGBM/embedding disagreement;
4. an LLM-reviewed result;
5. a difficult or conflict-prone result.

Selection uses fixed category precedence and stable offer/candidate tie-breaks.
If a category is empty, the next deterministic eligible category fills the
slot. Both the original targeted slot and the fallback reason are persisted.
The same offer cannot occur twice because the service deduplicates before
selection and the database enforces `UNIQUE(session_id, offer_id)`.

Each question is:

> Is this suggested SKU match correct?

The question includes all retained supplied candidates. A True answer confirms
the suggested candidate. A False answer is accepted only with exactly one of:

- a corrected SKU from those supplied candidates;
- `none_of_candidates`;
- `cannot_determine`.

Plain False is invalid. A response cannot be changed or submitted twice.
Database constraints and a conditional update protect against concurrent
duplicate answers.

## Label trust and training eligibility

Trust levels are:

- `GOLD`: decisive human confirmation, correction, or confirmation that none
  of the supplied candidates is valid.
- `SILVER`: structured LLM acceptance that passed Phase 6 policy. It remains
  review-governed and is not automatically included.
- `PSEUDO`: model-policy output without human confirmation. It is always
  `NOT_TRAINING_ELIGIBLE`.
- `REJECTED`: invalid, conflicting, inconclusive, or non-label output,
  including `cannot_determine`.

The store rejects any attempt to mark a PSEUDO label as automatically
eligible. Phase 7A prospective manifests include only answered GOLD review
IDs. Phase 7C materialized snapshots may additionally include
policy-qualified SILVER labels only when explicitly enabled and down-weighted.
Their records include artifact hashes, feature-schema identity, inclusion and
weight policy, override evidence, recent-GOLD evaluation identity, and
challenge exclusion proof. PSEUDO remains excluded.

## Sealed challenge-set isolation

Prospective training dataset registration accepts explicit sealed challenge
manifest paths. Every `SEALED_UNOPENED` manifest is hashed and its review
identities are collected. Any overlap with included review IDs aborts dataset
registration. The persisted proof records:

- checked manifest paths and hashes;
- sealed review IDs checked;
- an empty intersection;
- explicit challenge-row exclusion.

The existing Phase 6 challenge loader guard remains authoritative. Phase 7A
does not open or evaluate a sealed challenge set.

## Operational safety

Learning-store writes run through a non-blocking observation adapter.
Persistence failure is logged and cannot crash or change production inference.
Disabled ML mode does not invoke unified inference and therefore creates no
learning observation. Shadow and assisted decisions retain their Phase 6
production-isolation rules.

SQLite foreign keys, uniqueness constraints, WAL-compatible locking, stable
identifiers, and transactional writes protect referential integrity. The
database should be backed up with its SQLite-safe backup procedure before
manual maintenance.

## CLI usage

Inspect schema/counts:

```powershell
.\.venv\Scripts\python.exe scripts/inspect_learning_store.py
```

Export all completed human reviews:

```powershell
.\.venv\Scripts\python.exe scripts/export_learning_store.py
```

Export a specific table:

```powershell
.\.venv\Scripts\python.exe scripts/export_learning_store.py `
  --table pipeline_runs `
  --output data/learning/exports/pipeline_runs.csv
```

CSV exports use UTF-8-SIG for transparent Excel-compatible inspection. They
are exports, not authoritative mutable inputs.
