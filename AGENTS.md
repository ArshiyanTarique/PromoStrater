# AGENTS.md

# SKU Mapping ML Pipeline – Codex Operating Guide

This repository contains a production-oriented SKU Mapping pipeline for Salesflo.

The authoritative implementation is:

    sku_mapping_pipeline_ml.py

The long-term architecture and requirements are documented in:

    docs/ML_PIPELINE_SPEC.md

Read both before making any architectural decisions.

---

# PRIMARY OBJECTIVE

Transform the existing monolithic pipeline into a modular, production-grade
machine-learning pipeline while preserving existing business behaviour.

This is primarily a refactoring project.

Do not redesign working algorithms unless explicitly instructed.

Business correctness is more important than code elegance.

---

# PROJECT PRINCIPLES

Always prefer:

- correctness
- maintainability
- reproducibility
- deterministic behaviour
- backward compatibility
- explicit validation
- modularity
- performance

over unnecessary abstraction.

---

# AUTHORITATIVE FILES

Authoritative production pipeline:

    sku_mapping_pipeline_ml.py

Project specification:

    docs/ML_PIPELINE_SPEC.md

Current datasets:

    Alkabeer_Export_Data_Clickflyer.csv
    Product_Master.xlsx
    TrainingData/GOLD_TRAINING_PAIRS_v5_FINAL.csv

Never modify these datasets.

Read them when required.

Do not overwrite them.

---

# AUTONOMY

You are expected to work autonomously.

Do not stop simply to ask permission for normal engineering work.

You are authorised to:

- inspect repository files
- create new files
- modify repository files
- rename repository files
- delete files created during the refactor
- create folders
- install Python packages inside .venv
- run pytest
- run Python scripts
- run PowerShell commands
- run Git status
- create local commits
- generate reports
- generate processed datasets
- generate models
- regenerate outputs
- rerun failed commands
- fix implementation defects discovered during testing

If a command fails:

1. inspect the failure
2. identify the cause
3. apply the smallest correct fix
4. rerun
5. continue

Do not repeatedly ask for approval for ordinary repository work.

---

# DO NOT TOUCH OUTSIDE THIS REPOSITORY

Only work inside the currently opened repository.

Do NOT:

- inspect unrelated folders
- inspect Downloads
- inspect Documents
- inspect Desktop
- inspect browser data
- inspect passwords
- inspect SSH keys
- inspect API keys outside this repo
- inspect Windows settings
- inspect other Git repositories

Never intentionally access anything outside this repository.

---

# GIT RULES

Never:

git reset --hard

Never:

git clean -fd

Never rewrite Git history.

Never force push.

Never push to GitHub.

Always preserve unrelated user work.

Before each phase:

run

    git status

Understand the current changes before editing.

---

# CODING STYLE

Prefer:

- readable code
- small functions
- explicit names
- type hints
- pathlib
- dataclasses where appropriate
- logging
- unit tests

Avoid:

- unnecessary inheritance
- unnecessary wrappers
- deeply nested classes
- magic globals
- hidden side effects

---

# FEATURE ENGINEERING

The feature generator is the heart of the project.

Training and inference MUST use the exact same implementation.

Never duplicate feature generation logic.

Never allow feature drift.

The public API should eventually become:

    build_feature_vector()

and

    build_feature_vector_from_text()

---

# MODEL

The LightGBM model must always use exactly:

MODEL_FEATURE_COLUMNS

Never reorder them.

Never silently add features.

Never silently remove features.

Always validate model compatibility.

---

# TRAINING

Training must never leak data.

Offer groups must never cross train/test boundaries.

Synthetic rows must use:

build_feature_vector_from_text()

Real rows should use:

build_feature_vector()

when possible.

---

# CANDIDATE GENERATION

RapidFuzz candidate generation remains before ML.

Do not replace batch scoring with nested Python loops.

Maintain performance.

---

# COMPETITOR DISCOVERY

Competitor discovery only runs after a confirmed own-SKU match.

Never run competitor discovery for:

- manual review
- no match
- competitor-brand rows

---

# PERFORMANCE

The production pipeline should comfortably process hundreds of thousands of
rows.

Avoid unnecessary DataFrame copies.

Avoid O(N²) loops over the master catalogue.

Prefer vectorised operations.

Preserve RapidFuzz process.cdist.

---

# IMPORT SAFETY

Importing a module must never:

- load CSV files
- load Excel files
- load the model
- run inference
- start the pipeline

Imports should only define code.

---

# TESTING

Every significant refactor should include tests.

When implementing a phase:

run:

    python -m pytest

If tests fail:

diagnose

fix

rerun

Repeat until the relevant tests pass.

Never remove tests merely to make them pass.

---

# DATA SAFETY

Never overwrite:

Alkabeer_Export_Data_Clickflyer.csv

Product_Master.xlsx

TrainingData/GOLD_TRAINING_PAIRS_v5_FINAL.csv

Generated outputs belong in:

outputs/

or

data/processed/

Do not overwrite user data unless explicitly required.

---

# MODEL SAFETY

Generated models belong under:

models/

Never overwrite an existing model without creating a new version.

Always include metadata.

Always preserve feature ordering.

---

# WHEN TO STOP

Only stop and ask for clarification when:

1. requirements directly contradict one another

2. required files are genuinely missing

3. required data cannot be inferred safely

4. continuing would permanently destroy user data

5. external credentials are required

Otherwise continue autonomously.

---

# PHASE DISCIPLINE

Only implement the currently requested phase.

Do not begin future phases automatically.

Do not create speculative architecture.

Do not partially implement later milestones.

Finish the current phase completely before moving on.

---

# OUTPUT AFTER EACH PHASE

After finishing a phase provide:

1. files created

2. files modified

3. functions moved

4. tests run

5. test results

6. behavioural differences

7. known risks

8. recommended next phase

---

# QUALITY STANDARD

Code should be good enough that another ML engineer can immediately understand:

- how training works

- how inference works

- how features are generated

- how candidates are produced

- how decisions are made

without reading the original monolithic script.

---

# GOLDEN RULE

If unsure whether to preserve existing behaviour or invent a new behaviour:

Preserve the existing behaviour.

Only improve behaviour when explicitly instructed or when fixing a confirmed
bug.