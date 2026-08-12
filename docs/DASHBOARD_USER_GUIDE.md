# SKU Mapping Dashboard User Guide

## Purpose and safety boundary

The Streamlit dashboard provides a local interface for:

1. uploading and validating a ClickFlyer dump;
2. running the existing candidate, LightGBM, embedding, agreement, and
   optional LLM-review pipeline;
3. answering five targeted human-validation questions;
4. downloading validated run-scoped results.

The dashboard does not train or retrain a model. It cannot update the Product
Master, model weights, model registry, sealed challenge sets, or source
uploads. Model-only labels remain non-training-eligible.

The assisted operational threshold is 0.85. It is user-configured and is not
independently approved for automatic production matching. High scores or
model agreement do not prove that a mapping is correct.

## Installation and startup

Install the project and dashboard dependency:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dashboard]"
```

Start Streamlit from the repository root:

```powershell
.\.venv\Scripts\streamlit.exe run dashboard/Dashboard.py
```

Equivalent module command:

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard/Dashboard.py
```

The dashboard is intended for a controlled local environment. Streamlit's
configured maximum upload size is 100 MB; the application independently
enforces the same configurable limit.

## Page 1: Upload and Process

Accepted uploads are `.csv`, `.xlsx`, and `.xls` ClickFlyer dumps. The
application:

- removes directory components and unsafe characters from the filename;
- validates extension, content signature, size, and required columns;
- calculates SHA-256 over the exact uploaded bytes;
- checks SQLite for an active or completed run with the same hash;
- requires an explicit confirmation before identical bytes may be reprocessed;
- stages bytes atomically under a generated run-specific directory;
- reads tabular data without executing uploaded content.

Select either:

- `shadow`: model decisions and questions are observational; mappings are not
  eligible for competitor discovery.
- `assisted`: applies the explicit Phase 6 routing policy. This does not imply
  that the selected shadow-only model package is approved for production.

Only model IDs read from the safe registry are shown. Before processing, the
selected package is resolved through the registry and compatibility validator.
Free-text or filesystem model paths are not accepted.

Embedding and local LLM review remain disabled unless explicitly enabled.
Provider or model failures use the existing safe routes.

The progress display covers validation, reading, cleaning, candidate
generation, LightGBM, embeddings, agreement, LLM review, final decisions,
mapping export, competitor export, and question preparation. Failures produce
a short user-safe message. The internal traceback is stored in the run
directory and is never displayed in the dashboard.

Streamlit reruns do not automatically restart processing. Processing begins
only from the explicit button, and the file-hash check remains authoritative
after a browser refresh.

## Page 2: Human Validation

Select any persisted run with a review session. The database, not browser
state, supplies the questions and saved answers.

Each of the five questions displays:

- offer description;
- proposed SKU and master description;
- LightGBM confidence;
- decision source;
- LightGBM/embedding agreement;
- selection/fallback reasons and conflict flags;
- all retained top-candidate alternatives and their scores.

The question is:

> Is this suggested SKU match correct?

For True, save the response directly. It becomes a GOLD confirmation.

For False, select exactly one:

- a corrected SKU from the supplied candidates;
- None of these candidates;
- Cannot determine.

Plain False is not accepted. “Cannot determine” is stored as REJECTED, not
GOLD. Saved answers are immutable and duplicate submissions are blocked.
Back and Next may be used to inspect all questions, including saved answers.

Completion is displayed from 0/5 through 5/5. At 5/5, the answers are saved
for possible future controlled retraining. No retraining starts from the
dashboard.

## Page 3: Results and Downloads

Select any persisted run. The summary shows:

- input rows and unique own-brand offers;
- AUTO_ACCEPT, LLM_ACCEPT, MANUAL_REVIEW, NO_CANDIDATE, and MODEL_ERROR
  counts;
- model identities and runtime;
- a generic error indicator, when applicable.

Available downloads are:

- `sku_mapping_<run_id>.csv`
- `competitor_offers_<run_id>.csv`
- `run_summary_<run_id>.json`
- monitoring report, when one exists and is valid

The mapping CSV preserves the Phase 6E unified-decision columns. The
competitor CSV preserves:

1. `Al Kabeer Master SKU`
2. `Competitor Brand Names`
3. `Competitor Offers`

Competitor discovery receives only assisted-mode mappings explicitly marked
eligible by the final policy. Shadow/manual-review rows cannot enter the
competitor output.

The page offers a file only when it exists inside an approved output root,
is non-empty, and passes its header or JSON validation. Absolute server paths
are never displayed.

## Page 4: Models and Learning

The Models and Learning page is intentionally read-only. It displays:

- registered safe model choices;
- observed model-version records;
- run, prediction, GOLD-label, and unanswered-review counts.

Detailed learning visualization is deferred to Phase 7D. There are no
training, activation, retirement, or registry-edit controls.

## Persistence and recovery

The configured default database is:

`data/learning/sku_learning.db`

It stores run status, hashes, output metadata, candidate predictions, review
sessions, and answers. A browser refresh or Streamlit process restart does not
erase completed work. Use the run selector on validation or results pages to
resume.

Uploads and outputs are runtime data and are excluded from source control.
Back up SQLite using a SQLite-safe backup procedure before operational
maintenance.

## Troubleshooting

- **Duplicate blocked:** select the previous run, or check the explicit
  reprocessing confirmation after verifying that a new run is intended.
- **Missing columns:** correct the source export. The dashboard will not infer
  or silently rename required business columns.
- **Model unavailable:** confirm that the configured registry and immutable
  package exist. Arbitrary package paths are deliberately unsupported.
- **No five-question session:** at least five unique eligible offers with
  retained candidates are required.
- **No download button:** the artifact is missing, outside an approved root,
  empty, or failed schema validation.
- **Embedding/LLM unavailable:** the pipeline safely routes affected rows to
  review; it does not report false agreement or crash production processing.

For terminal commands and recovery checks, see
[Troubleshooting](TROUBLESHOOTING.md).
