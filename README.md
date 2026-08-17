# PromoStrater SKU Mapping

PromoStrater is:

> A monitored, human-in-the-loop SKU matching system that uses LightGBM,
> embeddings, deterministic rules and structured LLM review, collects five
> targeted human confirmations per upload, and improves through controlled
> champion–challenger retraining.

The repository is not approved for automatic production matching. The
assisted threshold of `0.85` is user-configured, not an accuracy claim or an
independently approved production threshold. LightGBM/embedding agreement and
LLM output are review evidence, not ground truth.

## What the system does

An uploaded ClickFlyer dump is validated and processed through:

1. existing RapidFuzz candidate generation (category-gated, top-K);
2. the shared feature contract — 19 base columns from
   `features/feature_generator.py`, plus 6 discriminative columns and 16
   group-relative rank columns for ranked models. The registered package
   declares its own list; `ranked-v5-cal` uses all 41 and sets
   `requires_group_features`, so a candidate shortlist must be featurised
   together rather than one pair at a time;
3. registered, calibrated LightGBM candidate scoring;
4. independent local embedding scoring;
5. deterministic agreement and hard-conflict policy;
6. optional structured local LLM review of supplied candidates only;
7. explicit `AUTO_ACCEPT`, `LLM_ACCEPT`, or safe review/failure outcomes;
8. five targeted True/False human questions when five eligible offers exist;
9. validated run-scoped SKU mapping and competitor CSV exports.

Inference never retrains a model. Human and automated observations are stored
in a schema-versioned SQLite learning store. Retraining is a separate,
operator-triggered champion–challenger workflow with evidence gates, immutable
snapshots, conservative comparison, explicit activation, and rollback.

## Run the dashboard on Windows

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard]"
.\.venv\Scripts\python.exe -m streamlit run dashboard/Dashboard.py
```

Open the local URL printed by Streamlit, normally
`http://localhost:8501`. Keep the terminal open to retain operational logs.
The dashboard is intended for a controlled local or trusted-network
environment; repository-defined authentication is not included.

The dashboard:

- accepts validated CSV, XLSX, or XLS uploads up to the configured limit;
- prevents accidental duplicate processing by exact file hash;
- lists registry-constrained model IDs only;
- persists runs and review answers in SQLite;
- survives browser refreshes;
- never starts retraining.

See [Dashboard User Guide](docs/DASHBOARD_USER_GUIDE.md) and
[Troubleshooting](docs/TROUBLESHOOTING.md).

## Deployment modes

| Mode | Behavior |
| --- | --- |
| `disabled` | Default. Modular assisted inference is not run. |
| `shadow` | Scores and records diagnostics; production-owned rows remain unchanged. |
| `assisted` | Applies the explicit agreement/LLM/hard-conflict policy and allows only explicitly eligible mappings into competitor discovery. |

Embedding and LLM review are separately disabled by default. Their failure
states are explicit and fail closed. See
[Deployment Modes](docs/DEPLOYMENT_MODES.md).

## Installation and verification

Create a virtual environment and install the project:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test,dashboard]"
```

The optional local sentence-transformer backend can be installed with:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[embedding]"
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run the bounded, isolated release audit:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_release_audit.py
```

Its five answers are fixture-only and are written to an isolated database
under `outputs/release_audit`; they are not real human evidence.

## Configuration

The typed configuration is loaded from `config/default.yaml`. Paths are
resolved relative to that file. Important defaults are:

- `ml.mode: disabled`
- `ml.auto_accept_threshold: 0.85`
- `embedding.enabled: false`
- `llm_review.enabled: false`
- `learning_store.database_path: ../data/learning/sku_learning.db`
- `retraining.minimum_new_gold_labels: 50`

Do not place credentials in the configuration file. The default LLM provider
is a local Ollama-compatible endpoint and accepts no embedded endpoint
credentials. See [Configuration Guide](docs/CONFIGURATION.md).

## Controlled retraining

Normal upload processing never trains. Offline controls are available through:

```powershell
.\.venv\Scripts\python.exe -m sku_mapping.retraining --help
```

The workflow is:

```text
baseline + eligible GOLD (+ explicitly enabled weighted SILVER)
  -> immutable snapshot
  -> leakage-safe challenger training and calibration
  -> same-row champion/challenger comparison
  -> assisted-use registration only if every policy gate passes
  -> separate explicit activation
  -> rollback if required
```

PSEUDO labels are excluded by default, and sealed challenge artifacts are
blocked from routine training and evaluation. See
[Retraining and Promotion Policy](docs/RETRAINING_AND_PROMOTION_POLICY.md).

## Repository map

Entry point: **`dashboard/Dashboard.py`**. `dashboard/bootstrap.py` puts
`src/` on `sys.path` before anything else, so the working tree always wins
over an installed copy.

- `dashboard/` — Streamlit pages (1–6) and page-independent services.
  `services/processing_service.py` is the background worker that runs a whole
  upload end to end.
- `src/sku_mapping/data/` — loaders, validators, preprocessing, offer identity.
- `src/sku_mapping/matching/` — `candidate_generator.py` is the production
  retriever; `matcher.py` / `rule_engine.py` are the pre-ML rule path, still
  used by a parity test and a dry-run script.
- `src/sku_mapping/features/` — shared training/inference feature contract.
- `src/sku_mapping/inference/` — unified assisted inference and final policy.
- `src/sku_mapping/shadow/` — model package loading, scoring, sampling,
  monitoring.
- `src/sku_mapping/agreement/` — deterministic routing policy.
- `src/sku_mapping/competitors/` — competitor discovery, optional ML
  re-ranking, and the human review queue.
- `src/sku_mapping/embedding/` — independent local embedding scorer and cache.
- `src/sku_mapping/llm_review/` — bounded provider interface, parser, policy, and cache.
- `src/sku_mapping/exports/` — business and run outputs.
- `src/sku_mapping/learning/` — SQLite store plus the migration history.
- `src/sku_mapping/ml/` — offline training, calibration, evaluation, leakage
  control, provenance.
- `src/sku_mapping/retraining/`, `src/sku_mapping/training/` — offline
  snapshots, challengers, comparison, activation, rollback, feature building.
- `models/` — registered packages and the registry. Tracked on purpose: a
  clone without them cannot score.
- `config/default.yaml` — the only configuration file.
- `tests/` — unit and integration suites; `scripts/` — offline CLI tools.
- `sku_mapping_pipeline_ml.py` — authoritative legacy production entry point
  retained for compatibility.

The dashboard and modular services do not import the legacy monolith because
that script intentionally executes when run. Core domain modules do not
depend on Streamlit.

**`src/sku_mapping/ml/calibration.py` must not be moved or deleted.** It is
not imported by the dashboard, but the registered joblib package stores
`IsotonicScoreCalibrator` and `ShadowModelPredictor` by module path, so
joblib resolves it at load time. Removing it breaks every model load.

## Competitor discovery

Rules decide membership; the model only orders the result.

```text
competitor offers
  -> category gate -> family / semantic-family overlap
  -> protein conflict -> form conflict -> pack conflict
  -> RapidFuzz score floors (raw 60 / adjusted 65)
  -> [optional] LightGBM re-ranking (raw margin, ordering only)
  -> per-target limit -> aggregate + long-format audit
  -> [optional] review queue -> Competitor Review page
```

The re-ranker reuses the registered own-brand package. Its score is a raw
margin comparable only inside one offer's shortlist; it is never read as a
competitor probability and never crosses a threshold. Both the re-ranker and
the review queue are off by default:

- `competitors.ml_reranking_enabled: false`
- `competitors.ml_shortlist_top_k: 0` (0 adopts the package's `retrieval_k`)
- `competitors.brand_stripping_enabled: true`
- `competitors.review_staging_per_target: 0`

There is no competitor-labelled ground truth yet, so no precision or recall
figure exists for competitor discovery — from the rules or the model. The
review queue exists to start collecting it.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Configuration](docs/CONFIGURATION.md)
- [Deployment Modes](docs/DEPLOYMENT_MODES.md)
- [Dashboard User Guide](docs/DASHBOARD_USER_GUIDE.md)
- [Learning Data Governance](docs/LEARNING_DATA_GOVERNANCE.md)
- [LLM Review Policy](docs/LLM_REVIEW_POLICY.md)
- [Retraining and Promotion](docs/RETRAINING_AND_PROMOTION_POLICY.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Pipeline Specification](docs/ML_PIPELINE_SPEC.md)

## Current limitations

- The registered production package is `ranked-v5-cal-20260810T100756Z-matcher`
  (`models/registry/matcher_ranked_v5_calibrated.joblib`). It is
  `SHADOW_MODE_ONLY` and `NOT_APPROVED_FOR_AUTOMATIC_MATCHING`; its own
  thresholds are auto `0.99` / review `0.94`, fitted on own-brand data only.
  The separate `0.85` in configuration is a pipeline setting, not that model's
  threshold.
- The real learning store currently has no GOLD labels; the default 50-label
  retraining gate therefore blocks a real snapshot.
- The audited default embedding encoder is deterministic lexical hashing, not
  a learned semantic model. It is approved for offline retrieval expansion
  and compatible-candidate ranking support, not independent auto-match
  authority. The optional sentence-transformer remains unavailable until a
  revision-pinned model and dependency are installed locally and evaluated.
- SQLite is intended for a single-host controlled deployment.
- Streamlit authentication and multi-user authorization are operational
  deployment responsibilities.
- The sealed challenge set has not been opened or evaluated.
- No model-improvement claim is supported until a challenger passes governed
  evaluation on sufficient real human-reviewed evidence.
