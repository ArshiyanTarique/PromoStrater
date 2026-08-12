# Troubleshooting

## Streamlit shows a redacted application error

Run Streamlit from the repository root and keep the terminal visible:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/Dashboard.py
```

The browser intentionally hides stack traces and server paths. Read the
terminal output or the controlled run failure log. Do not paste secrets or
unredacted customer data into an issue.

Verify the environment:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -c "import streamlit; print(streamlit.__version__)"
.\.venv\Scripts\python.exe -m pytest -q tests\integration\test_dashboard_streamlit_smoke.py
```

If imports fail, reinstall the dashboard extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard]"
```

## Model is unavailable

- Select only an ID shown by the dashboard.
- Confirm `models/model_registry.json` exists and keeps automatic production
  matching disabled.
- Confirm the registered package exists under `models/registry`.
- Run the package tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\unit\test_model_package.py tests\unit\test_shadow_predictor.py
```

An invalid package must remain a `MODEL_ERROR`/safe fallback. Do not bypass the
registry or compatibility validator.

## Embedding model is unavailable

Embedding is optional and disabled by default. Install the local backend extra
only when explicitly required:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[embedding]"
```

Pin model provenance before relying on persistent cache reuse. Failure is
expected to route safely and must not be described as agreement.

## Ollama review is unavailable

LLM review is disabled by default. If it is explicitly enabled, verify the
configured local endpoint and model outside customer processing. Provider
failure, timeout, malformed output, or an arbitrary candidate ID routes to
manual review. Do not change `fail_route` away from the supported safe policy.

## Upload is rejected

Check:

- extension is CSV, XLSX, or XLS;
- file is within the configured size limit;
- content signature matches the extension;
- required ClickFlyer columns are present;
- the file contains at least one supported own-brand offer.

The application does not execute uploaded content and will not silently rename
or infer missing business columns.

## Duplicate processing is blocked

The exact file bytes already have an active or completed run. Select the
persisted run from the validation/results page. Use the explicit duplicate
confirmation only when a second processing record is intended.

## No five-question session appears

A session is created only after a successful persisted run with at least five
eligible unique offers. Missing target categories use recorded fallbacks, but
fewer than five eligible offers produces no session.

## A False answer cannot be saved

False requires exactly one correction:

- choose a supplied candidate;
- choose None of these candidates; or
- choose Cannot determine.

Cannot determine is REJECTED rather than GOLD. Saved answers are immutable.

## A download button is missing

The dashboard shows an artifact only if it:

- exists under an approved output root;
- is non-empty;
- has the exact expected CSV header or valid JSON.

Inspect the persisted run status and safe error summary. Do not expose or
manually serve an artifact that failed validation.

## SQLite is locked or unavailable

The repository uses SQLite for a single-host workflow. Avoid multiple
independent hosts writing the same database. Keep the database on a reliable
local filesystem and use a SQLite-safe backup procedure. A multi-host
deployment requires a server database and equivalent constraints.

Inspect non-sensitive counts:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_learning_store.py
```

## Retraining is blocked

This is expected until the configured minimum number of new real GOLD labels
exists. The current default is 50:

```powershell
.\.venv\Scripts\python.exe -m sku_mapping.retraining build-training-snapshot --config config/default.yaml
```

A development override must include an explicit reason and is not production
evidence. Never point routine training or comparison at a sealed challenge
artifact.

## Release verification

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_final_release_audit.py
.\.venv\Scripts\python.exe -m pytest
git diff --check
git status --short
```

The bounded audit uses an isolated database and fixture reviewer identity. Its
five GOLD rows are test fixtures, not real human-reviewed evidence.
