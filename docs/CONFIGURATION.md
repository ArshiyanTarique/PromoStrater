# Configuration Guide

## Loading and path behavior

The typed configuration is defined in `config/default.yaml` and loaded by
`sku_mapping.config.load_config`. Relative paths are resolved against the
configuration file directory, not the caller's current directory.

Repository discovery is centralized in `sku_mapping.paths` and is based on
the installed source location, not the process working directory. The default
CLI paths therefore remain inside the active repository even when a command
is launched from another directory. Do not put a machine-specific repository
root in configuration.

The repository-owned defaults resolve as follows:

| Runtime content | Default repository-relative location |
|---|---|
| Processed and training data | `data/processed/` |
| Dashboard uploads | `data/dashboard_uploads/` |
| Learning database and exports | `data/learning/` |
| Review staging and challenge sets | `data/review_staging/`, `data/challenge_sets/` |
| Production and dashboard outputs | `outputs/`, `outputs/dashboard_runs/` |
| Shadow outputs | `outputs/shadow/` |
| Models, registry, and metadata | `models/` |
| Training and comparison reports | `reports/` |
| Embedding and LLM caches | `data/processed/` |

Absolute paths remain supported as explicit operator overrides. Such an
override is intentionally external and is not automatically relocated with
the repository. Atomic temporary files are created beside their destination,
so repository-owned writes do not use an unrelated system temporary folder.

Keep credentials out of this file. `.streamlit/secrets.toml` is ignored by
Git, but the current application does not require a paid API key.

## Safe defaults

```yaml
ml:
  mode: disabled
  model_id: null
  auto_accept_threshold: 0.85
  require_registered_model: true
  apply_safety_overrides: true

embedding:
  enabled: false
  backend: local_hashing
  model_name: sku-hashing-384
  model_version: sklearn-hashing-word-1-2-384-v2
  device: cpu
  similarity_metric: cosine
  cache_embeddings: true
  local_files_only: true
  text_construction_version: 2.0.0
  commercial_parser_version: 1.0.0
  retrieval_enabled: true
  retrieval_top_k: 5

llm_review:
  enabled: false
  provider: ollama
  endpoint: http://localhost:11434
  temperature: 0
  minimum_accept_confidence: 0.85
  fail_route: manual_review
```

The two `0.85` values have different meanings:

- `ml.auto_accept_threshold` is the user-configured calibrated LightGBM
  operational boundary used by agreement routing.
- `llm_review.minimum_accept_confidence` is a minimum on the provider's
  self-reported confidence. It is not calibrated probability.

Neither value is an accuracy guarantee or production approval.

## Deployment selection

For a controlled assisted configuration:

```yaml
ml:
  mode: assisted
  model_id: alkabeer-sku-matcher-v3-20260729T061802974421Z-8c636b0ac4a2
  auto_accept_threshold: 0.85
  require_registered_model: true
  apply_safety_overrides: true
```

The model ID must exist exactly once in `models/model_registry.json`, resolve
inside the configured registry directory, and pass package validation. There
is no newest-model fallback and no arbitrary dashboard model path.

## Embeddings

Audited, dependency-free offline backend:

```yaml
embedding:
  enabled: true
  backend: local_hashing
  model_name: sku-hashing-384
  model_version: sklearn-hashing-word-1-2-384-v2
  batch_size: 64
  device: cpu
  similarity_metric: cosine
  cache_embeddings: true
  cache_path: ../data/processed/embedding_cache.sqlite3
  max_sequence_length: 256
  pooling_strategy: mean
  normalize_vectors: true
  local_files_only: true
  text_construction_version: 2.0.0
  commercial_parser_version: 1.0.0
  retrieval_enabled: true
  retrieval_top_k: 5
  retrieval_offer_batch_size: 256
```

This backend is a deterministic lexical hashing encoder. It is approved for
bounded retrieval expansion and ranking support within a commercial
compatibility class; it is not a learned semantic model and is not approved
to override commercial conflicts or independently authorize auto-matches.
Accordingly, `agreement.allow_embedding_auto_accept` defaults to `false`.
Changing that separate safety approval requires a reviewed precision study;
it does not alter the frozen LightGBM threshold.

The optional sentence-transformer backend remains available when an approved,
revision-pinned model has already been installed locally:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[embedding]"
```

Use `backend: local_sentence_transformer`, an explicit `model_version`, and
`local_files_only: true`. Normal production must not rely on an implicit
download. A load failure is reported and leaves fuzzy/LightGBM processing
available; it never silently switches models.

Cache reuse requires exact normalized text, model ID, and model version. If a
safe model version cannot be resolved, caching fails closed.

## Structured local LLM review

```yaml
llm_review:
  enabled: true
  provider: ollama
  model: llama3.1:8b
  endpoint: http://localhost:11434
  timeout_seconds: 60
  maximum_candidates: 5
  temperature: 0
  minimum_accept_confidence: 0.85
  maximum_retries: 1
  fail_route: manual_review
  reject_all_route: manual_review
```

Endpoints may use absolute HTTP or HTTPS URLs, but embedded credentials,
queries, and fragments are rejected. The reviewer sends one bounded offer and
at most the configured candidate count; it never sends the entire Product
Master.

Failure, timeout, malformed output, an arbitrary SKU ID, low confidence, or a
hard deterministic conflict routes safely to manual review.

## Dashboard and learning store

```yaml
learning_store:
  enabled: true
  database_path: ../data/learning/sku_learning.db
  csv_export_directory: ../data/learning/exports
  questions_per_run: 5

dashboard:
  input_directory: ../data/dashboard_uploads
  output_directory: ../outputs/dashboard_runs
  max_upload_size_mb: 100
  allowed_extensions: [csv, xlsx, xls]
```

The dashboard and `.streamlit/config.toml` enforce the 100 MB default. Runtime
uploads, outputs, databases, caches, reports, models, and local Streamlit
secrets are ignored by Git.

## Retraining policy

```yaml
retraining:
  minimum_new_gold_labels: 50
  recent_gold_holdout_count: 20
  include_silver: false
  gold_weight: 1.0
  silver_weight: 0.25
  pseudo_weight: 0.0
  operational_threshold: 0.85
```

A development override requires an explicit minimum and recorded reason.
Production-like operation should retain the 50-GOLD default. Normal inference
never reads this section to train a model.

## Validate configuration

```powershell
.\.venv\Scripts\python.exe -c "from sku_mapping.config import load_config; c=load_config('config/default.yaml'); print(c.ml.mode, c.ml.auto_accept_threshold)"
```

Configuration validation rejects invalid threshold order, unsupported routes,
unsafe extensions, unreasonable upload sizes, and invalid provider settings.
