You are a senior machine-learning engineer and Python software architect.

I have an existing SKU-matching pipeline in a single Python file named:

sku_mapping_pipeline_ml.py

Treat this file as the authoritative starting point.

The current script already performs:

1. ClickFlyer data loading.
2. Product master loading.
3. Own-brand detection.
4. Text cleaning and normalisation.
5. Weight and volume parsing.
6. Multipack parsing.
7. Category detection.
8. Product-family normalisation.
9. RapidFuzz candidate generation.
10. Rule-based confidence assignment.
11. LightGBM candidate review.
12. Competitor discovery.
13. Final CSV generation.

The current file works, but it is monolithic, tightly coupled, difficult to test, and mixes:

- configuration,
- data loading,
- feature engineering,
- candidate generation,
- model inference,
- decision thresholds,
- competitor discovery,
- validation,
- exporting,
- and execution logic

inside one script.

The objective is to refactor this into a maintainable, corporate-level machine-learning pipeline that supports:

- reproducible training,
- identical feature generation during training and inference,
- modular testing,
- configuration management,
- model versioning,
- logging,
- data validation,
- threshold tuning,
- batch inference,
- reproducible exports,
- and safe future extension.

Do not throw away the existing business rules.

Do not rewrite the matching logic from scratch unless necessary.

Preserve the current behaviour as much as possible while restructuring it.

==================================================
CURRENT IMPORTANT LOGIC
==================================================

The existing script contains the feature function:

_build_ml_feature_row(offer_row, master_row)

It currently produces these 19 model features:

1. protein_match
2. family_match
3. variant_match
4. size_match
5. pack_format_match
6. word_similarity
7. character_similarity
8. token_similarity
9. unit_pack_weight_g
10. number_of_units
11. bonus_weight_g
12. total_offer_weight_g
13. piece_count
14. master_unit_weight_g
15. master_units_per_carton
16. is_mixed_protein_offer
17. is_multi_family_offer
18. contains_non_meat_product
19. expected_match_count

These features must continue to exist unless a migration strategy is explicitly provided.

The current model package contains:

- model
- feature_columns
- auto_match_threshold
- manual_review_threshold
- model_version

The current production decision logic is:

probability >= auto_match_threshold
    -> AUTO_MATCH

probability >= manual_review_threshold
    -> MANUAL_REVIEW

otherwise
    -> NO_MATCH

The current fuzzy candidate stage must remain before the ML review stage.

The intended production architecture is:

New offer
    -> normalisation and parsing
    -> candidate generation
    -> feature generation for offer-candidate pair
    -> LightGBM probability
    -> decision policy
    -> matched SKU / review / no match

The intended training architecture is:

Gold pair dataset
    -> feature generation using the exact same feature code
    -> group-aware train/validation/test split
    -> LightGBM training
    -> threshold tuning
    -> evaluation
    -> versioned model package

==================================================
PRIMARY REQUIREMENT
==================================================

Refactor the current script into a complete project with clear modules.

Use the following target structure as a guide:

sku_mapping_project/
│
├── config/
│   ├── default.yaml
│   └── logging.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── outputs/
│
├── models/
│   ├── registry/
│   └── metadata/
│
├── src/
│   └── sku_mapping/
│       ├── __init__.py
│       ├── config.py
│       ├── constants.py
│       ├── schemas.py
│       ├── logging_utils.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loaders.py
│       │   ├── validators.py
│       │   └── preprocessing.py
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── text_features.py
│       │   ├── measurement_features.py
│       │   ├── semantic_features.py
│       │   └── feature_generator.py
│       │
│       ├── matching/
│       │   ├── __init__.py
│       │   ├── candidate_generator.py
│       │   ├── rule_engine.py
│       │   └── matcher.py
│       │
│       ├── ml/
│       │   ├── __init__.py
│       │   ├── model_loader.py
│       │   ├── predictor.py
│       │   ├── decision_policy.py
│       │   ├── trainer.py
│       │   ├── evaluator.py
│       │   └── threshold_tuning.py
│       │
│       ├── competitors/
│       │   ├── __init__.py
│       │   └── discovery.py
│       │
│       ├── exports/
│       │   ├── __init__.py
│       │   ├── report_builder.py
│       │   └── validators.py
│       │
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── inference_pipeline.py
│       │   └── training_pipeline.py
│       │
│       └── cli.py
│
├── scripts/
│   ├── run_inference.py
│   ├── build_training_features.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
│
├── notebooks/
│   └── model_experiments.ipynb
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── .gitignore

The exact structure may be simplified if justified, but responsibilities must remain separated.

==================================================
MILESTONE 1 — EXTRACT FEATURE ENGINEERING
==================================================

Create a reusable feature-generation package.

Move the relevant logic from the current script into dedicated modules.

This includes:

- clean_offer_text
- unit_dim_value
- extract_measures_detailed
- extract_flyer_measures
- extract_master_measures
- collapse_to_simple
- pack_is_compatible
- pack_structure_agrees
- _first_weight_value
- _offer_unit_count
- _offer_total_weight
- _extract_piece_count
- _extract_bonus_weight
- _master_units_per_carton
- _protein_set
- _family_set
- _variant_set
- _compatibility_flag
- _expected_match_count
- _build_ml_feature_row

Rename the public feature function to:

build_feature_vector

Required signature:

def build_feature_vector(
    offer_row: Mapping[str, Any],
    master_row: Mapping[str, Any],
) -> dict[str, float | int | None]:

Also provide:

def build_feature_vector_from_text(
    offer_text: str,
    master_row: Mapping[str, Any],
    product: str = "",
    variant: str = "",
    base_packsize: str = "",
) -> dict[str, float | int | None]:

This second function is required for synthetic gold-training rows that do not exist in the ClickFlyer dump.

The feature generator must:

1. Avoid loading CSV or Excel files.
2. Avoid loading an ML model.
3. Avoid making decisions such as AUTO_MATCH or NO_MATCH.
4. Avoid using global DataFrames.
5. Accept dictionaries, pandas Series, or mapping-compatible objects.
6. Generate all 19 features.
7. Return features in a deterministic order.
8. expose a constant:

MODEL_FEATURE_COLUMNS = [...]

9. Handle missing values safely.
10. Return numeric values or NaN, never unexpected strings.
11. Use the same parsing logic for both training and inference.
12. Include docstrings and type hints.
13. Include unit tests.

Add tests for at least:

- chicken offer vs chicken SKU,
- chicken offer vs beef SKU,
- matching 400g pack,
- mismatched 400g vs 1kg pack,
- 2 x 500g flyer pack,
- master case specification such as 270 Gms x 20 Pkts,
- bonus pack such as 750g + 250g,
- piece count such as 20 pcs,
- synthetic offer input,
- missing measurement data,
- volume not being compared to weight,
- mixed-protein offer.

Do not alter the meaning of the current measurement logic.

In particular:

- flyer multipacks may include both total and per-unit values,
- master carton totals must not be treated as retail pack sizes,
- weight and volume must remain separate dimensions.

==================================================
MILESTONE 2 — DATA LOADING AND VALIDATION
==================================================

Create proper data loaders for:

1. ClickFlyer CSV.
2. Product Master Excel.
3. Gold training-pairs CSV.
4. Saved model packages.

Use pathlib.Path instead of hard-coded path strings.

Configuration should define paths such as:

flyer_path
master_path
gold_pairs_path
output_dir
model_dir
model_path

Validate required columns before processing.

ClickFlyer required columns should include those currently used, such as:

- Offer Name
- Product
- Brand Name
- Variant
- Base Packsize
- Country
- Retailer Name
- Flyer Name
- offerid
- Offer Price
- Regular Price

Product Master required columns should include:

- Itemcode
- Itemname
- Item-Cat-2
- Item-Cat-4
- Item Description
- Item-Spec

Gold dataset required columns should include:

- offer_group_id
- offer_text
- master_itemcode
- pair_label
- use_for_binary_pair_training

Where available, also support:

- source_dataset
- product_class_offer
- variant_offer
- recommended_split
- split_group
- label_provenance
- label_confidence

Validation must fail early with clear errors.

Normalise Itemcode consistently.

Do not silently convert missing master codes into arbitrary rows.

If a gold row contains a master_itemcode absent from Product_Master.xlsx:

- record it in a rejected-rows report,
- do not train on it,
- do not guess a replacement.

==================================================
MILESTONE 3 — BUILD TRAINING FEATURES
==================================================

Create a script:

scripts/build_training_features.py

Inputs:

- GOLD_TRAINING_PAIRS_v5_FINAL.csv
- Product_Master.xlsx
- optionally ClickFlyer dump

Important clarification:

The master_itemcode comes from each gold pair row.

It does not come from the ClickFlyer dump.

For every gold pair:

1. Read offer_text.
2. Read master_itemcode.
3. Find the corresponding master row in Product_Master.xlsx.
4. Construct the offer-side representation.
5. Generate features with build_feature_vector or
   build_feature_vector_from_text.
6. Append:
   - pair_label
   - offer_group_id
   - record_id if present
   - source_dataset
   - label_provenance
   - recommended_split
7. Save the result as a processed training feature table.

Real flyer rows:

If a reliable match to the ClickFlyer dump can be established, recover:

- Product
- Variant
- Base Packsize
- original offer fields

However, matching to the dump must not be mandatory for training.

Synthetic rows:

Construct the offer-side input directly from:

- offer_text
- product_class_offer if available
- variant_offer if available

Do not discard synthetic rows simply because they do not appear in the dump.

For synthetic rows, the master SKU is still resolved using:

gold_row["master_itemcode"]

against Product_Master.xlsx.

Output files:

data/processed/training_features.parquet
data/processed/training_features.csv
data/processed/rejected_training_rows.csv
data/processed/training_feature_manifest.json

The manifest should contain:

- generated timestamp,
- source file names,
- row counts,
- accepted row count,
- rejected row count,
- class distribution,
- feature names,
- feature schema,
- dataset fingerprint or hash,
- source provenance distribution.

Only include rows where:

use_for_binary_pair_training == 1

For binary training, only allow:

pair_label == 0
pair_label == 1

Exclude abstain labels such as -1.

Log all exclusions.

==================================================
MILESTONE 4 — CANDIDATE GENERATION
==================================================

Extract Stage 2 fuzzy candidate generation into:

matching/candidate_generator.py

Preserve the current approach:

- hard category gating,
- RapidFuzz process.cdist,
- token_sort_ratio,
- token_set_ratio,
- pack compatibility adjustment,
- excluded known-incompatible packs,
- stricter handling for category Other,
- best-candidate selection,
- second-best margin,
- raw margin,
- structure conflict handling.

Create a CandidateMatch data structure with fields such as:

- itemcode
- itemname
- text_score
- adjusted_score
- margin
- raw_margin
- pack_status
- pack_structure_status
- category
- candidate_rank

The candidate generator should support:

generate_best_candidate(...)
generate_top_candidates(..., top_k=5)
generate_candidates_batch(...)

The ML model must review candidate pairs.

It must not search the entire master catalogue itself.

Production flow:

offer
    -> candidate generator
    -> top candidate or top-k candidates
    -> feature generator
    -> model
    -> decision policy

Do not confuse candidate generation with final acceptance.

==================================================
MILESTONE 5 — MODEL TRAINING
==================================================

Create a reproducible LightGBM training pipeline.

Use:

- lightgbm.LGBMClassifier
- GroupShuffleSplit or GroupKFold
- group by offer_group_id

The same offer_group_id must never appear across train and validation/test sets.

If recommended_split exists and is valid, support using it.

Otherwise perform a group-aware split.

Do not use an ordinary random row split.

Train only on the generated feature table.

Never train directly on raw offer_text columns unless a separate text model is deliberately added later.

Initial feature set must be exactly MODEL_FEATURE_COLUMNS.

Handle class imbalance using one of:

- class_weight="balanced",
- scale_pos_weight,
- or tuned sample weights.

Do not blindly optimise accuracy.

Report:

- ROC-AUC,
- PR-AUC,
- precision,
- recall,
- F1,
- confusion matrix,
- false-positive rate,
- false-negative rate,
- calibration information,
- metrics by source_dataset,
- metrics by label_provenance,
- metrics by product family if available.

Because wrong automatic matches are more damaging than manual reviews, threshold selection should prioritise high precision for AUTO_MATCH.

Create three datasets:

- train,
- validation,
- test.

Use validation for:

- hyperparameter tuning,
- early stopping,
- threshold selection.

Use test only once for final reporting.

Use a fixed random seed.

Store all random seeds in configuration.

==================================================
MILESTONE 6 — THRESHOLD TUNING
==================================================

Create:

ml/threshold_tuning.py

The system has three decisions:

AUTO_MATCH
MANUAL_REVIEW
NO_MATCH

Tune two thresholds:

auto_match_threshold
manual_review_threshold

Constraints:

auto_match_threshold > manual_review_threshold

Select thresholds based on business goals.

Recommended policy:

AUTO_MATCH:
- extremely high precision,
- low false-positive rate.

MANUAL_REVIEW:
- uncertain candidate,
- preserve recall without accepting automatically.

NO_MATCH:
- weak probability or unsuitable candidate.

Produce a threshold report containing:

- candidate threshold combinations,
- auto-match precision,
- auto-match recall,
- manual-review volume,
- no-match volume,
- false automatic matches,
- accepted coverage.

Do not hard-code 0.95 and 0.70 without evaluating them.

Allow them as defaults, but tune and save the chosen values.

==================================================
MILESTONE 7 — MODEL PACKAGE AND VERSIONING
==================================================

Save the model as a versioned package.

Example:

models/registry/alkabeer_sku_matcher_v2_2026-07-28.joblib

The package should contain:

{
    "model": trained_model,
    "feature_columns": MODEL_FEATURE_COLUMNS,
    "auto_match_threshold": ...,
    "manual_review_threshold": ...,
    "model_version": ...,
    "training_timestamp": ...,
    "training_dataset_hash": ...,
    "metrics": {...},
    "lightgbm_version": ...,
    "sklearn_version": ...,
    "python_version": ...,
    "feature_generator_version": ...,
    "training_config": {...}
}

Also write a human-readable metadata JSON.

Before inference, validate:

- all required model-package keys exist,
- feature columns match the current feature generator,
- model supports predict_proba,
- feature ordering matches exactly.

Fail loudly if the deployed feature generator and model disagree.

==================================================
MILESTONE 8 — INFERENCE PIPELINE
==================================================

Create:

pipelines/inference_pipeline.py

The inference pipeline should orchestrate:

1. Load configuration.
2. Load ClickFlyer data.
3. Load Product Master.
4. Validate schemas.
5. Clean and preprocess data.
6. Identify own-brand rows.
7. Generate fuzzy candidates.
8. Generate model features.
9. Run LightGBM prediction.
10. Apply decision policy.
11. Build final mappings.
12. Run competitor discovery only for accepted mappings.
13. Validate outputs.
14. Write reports.

Create a main service class such as:

class SKUMappingPipeline:
    def __init__(self, config: PipelineConfig):
        ...

    def run(self) -> PipelineResult:
        ...

Avoid module-level execution.

The pipeline must not run merely because a module was imported.

Use:

if __name__ == "__main__":
    ...

only in CLI scripts.

Support batch processing.

Keep vectorised and batched RapidFuzz scoring where currently used.

Do not replace efficient process.cdist calls with slow nested Python loops.

==================================================
MILESTONE 9 — DECISION POLICY
==================================================

Create a dedicated decision-policy module.

Example:

class MatchDecision(str, Enum):
    AUTO_MATCH = "AUTO_MATCH"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    NO_MATCH = "NO_MATCH"
    NO_CANDIDATE = "NO_CANDIDATE"
    MASTER_SKU_NOT_FOUND = "MASTER_SKU_NOT_FOUND"

Create:

def classify_probability(
    probability: float | None,
    auto_match_threshold: float,
    manual_review_threshold: float,
) -> MatchDecision:

Keep decision logic separate from feature generation and prediction.

The predictor should return a probability.

The decision policy should convert probability into an action.

==================================================
MILESTONE 10 — COMPETITOR DISCOVERY
==================================================

Extract Stage 3 into:

competitors/discovery.py

Preserve the existing business behaviour:

- use accepted own SKU as the target,
- scope by Country and normalised product family,
- compare competitor offers against target master text,
- use each flyer offer's own retail pack,
- exclude known pack mismatches,
- distinguish direct competitor from pack unverified,
- retain raw score,
- retain adjusted score,
- limit to top competitor brands.

Competitor discovery must only run after a reliable own-SKU mapping exists.

It must not run for:

- manual review,
- no match,
- no candidate,
- competitor-brand source rows.

==================================================
MILESTONE 11 — EXPORTS
==================================================

Preserve existing output requirements.

Produce:

1. Detailed mapping output.
2. Final master-SKU-to-ClickFlyer-offers output.
3. Final competitor-offers output.
4. Manual-review queue.
5. Run summary.
6. Rejected/error rows.

Detailed mapping output should include:

- original offer identifiers,
- original offer text,
- suggested itemcode,
- suggested itemname,
- final matched itemcode,
- final matched itemname,
- fuzzy score,
- adjusted score if available,
- margin,
- raw margin,
- pack status,
- pack structure status,
- ml_probability,
- ml_decision,
- model_version,
- feature-generator version,
- processing timestamp,
- candidate rank,
- reason code.

Do not store the entire feature vector as a JSON string by default in the business-facing output.

Feature vectors may be written to a separate diagnostics file.

Preserve the master order in final SKU reports.

Keep the CSV newline validation.

Ensure stable output schemas even when result sets are empty.

==================================================
MILESTONE 12 — LOGGING
==================================================

Replace print statements with the logging module.

Use structured, useful logging.

Log:

- run ID,
- file paths,
- row counts,
- stage durations,
- class counts,
- candidate counts,
- model version,
- feature count,
- decision counts,
- missing master SKUs,
- invalid rows,
- output paths.

Use log levels:

- INFO for normal stage progress,
- WARNING for recoverable data issues,
- ERROR for failed rows or invalid state,
- DEBUG for detailed diagnostics.

Do not log confidential raw data unnecessarily.

Create one log file per run.

==================================================
MILESTONE 13 — CONFIGURATION
==================================================

Move constants out of the main script.

Configuration should include:

data:
  flyer_path:
  master_path:
  gold_pairs_path:
  output_dir:

model:
  path:
  expected_feature_count:
  auto_match_threshold:
  manual_review_threshold:

matching:
  category_other_min_score:
  category_other_min_margin:
  normal_min_score:
  pack_compatible_bonus:
  pack_unknown_penalty:
  high_score_threshold:
  high_margin_threshold:
  medium_score_threshold:
  medium_margin_threshold:

competitors:
  raw_score_floor:
  adjusted_score_floor:
  max_per_target:

runtime:
  random_seed:
  log_level:
  output_encoding:

Use a typed configuration object.

Validate:

0 <= manual_review_threshold < auto_match_threshold <= 1

==================================================
MILESTONE 14 — SCHEMAS AND TYPES
==================================================

Use dataclasses, TypedDict, or Pydantic models where appropriate.

Recommended types:

OfferRecord
MasterSKURecord
CandidateMatch
FeatureVector
PredictionResult
MappingDecision
PipelineResult
ModelMetadata

Do not over-engineer every DataFrame row into a class if it harms batch performance.

Use schemas primarily at module boundaries.

==================================================
MILESTONE 15 — TESTING
==================================================

Create unit tests for:

- brand normalisation,
- own-brand aliases,
- category parsing,
- singular/plural family normalisation,
- measure parsing,
- flyer multipacks,
- master case packs,
- weight-vs-volume separation,
- pack compatibility,
- structure compatibility,
- protein detection,
- family detection,
- variant detection,
- all 19 generated features,
- missing values,
- model-package validation,
- decision thresholds,
- candidate ranking,
- group leakage prevention,
- export schemas.

Create integration tests for:

1. Tiny flyer fixture.
2. Tiny master fixture.
3. Tiny gold-pair fixture.
4. Fake trained model or mocked predictor.
5. Full inference flow.
6. Full training-feature generation flow.

Add regression tests using several known problematic examples:

- chicken offer incorrectly matching beef,
- chicken fillet incorrectly matching fish fillet,
- chicken seekh incorrectly matching beef seekh,
- 2x500g incorrectly matching plain 1kg where structure differs,
- master carton total incorrectly compared with retail offer,
- 1L incorrectly interpreted as 1kg.

==================================================
MILESTONE 16 — TRAINING DATA GOVERNANCE
==================================================

The gold dataset contains multiple provenance types.

Preserve provenance fields.

Do not treat every positive label as equally trustworthy.

Support optional sample weighting by:

- label_confidence,
- label_provenance,
- source_dataset.

Rule-generated likely-correct rows should not automatically become trusted positives unless explicitly approved.

Synthetic data must not leak between train and test through repeated offer templates.

Group by offer_group_id.

Also inspect near-duplicate synthetic templates.

Create a training-data audit report with:

- duplicate pairs,
- conflicting labels,
- same offer across splits,
- invalid labels,
- unknown master SKUs,
- class imbalance,
- provenance imbalance,
- source imbalance.

==================================================
MILESTONE 17 — CORPORATE-LEVEL SAFETY CHECKS
==================================================

Add the following safeguards:

1. Feature schema check before prediction.
2. Model metadata compatibility check.
3. Input schema validation.
4. Missing-SKU reporting.
5. Empty-data handling.
6. Stable output schemas.
7. No overwrite unless explicitly allowed.
8. Atomic model saving.
9. Atomic output writing where practical.
10. Run IDs.
11. Reproducible timestamps.
12. Dataset hashes.
13. Model-version tracking.
14. Threshold-version tracking.
15. Clear exception messages.
16. No hidden network calls.
17. No Ollama dependency in the final ML pipeline.
18. No paid API dependency.
19. No implicit global state.
20. No execution during imports.

==================================================
MILESTONE 18 — COMMAND-LINE INTERFACE
==================================================

Create CLI commands such as:

python -m sku_mapping.cli build-features \
    --gold data/raw/GOLD_TRAINING_PAIRS_v5_FINAL.csv \
    --master data/raw/Product_Master.xlsx \
    --output data/processed/training_features.parquet

python -m sku_mapping.cli train \
    --features data/processed/training_features.parquet \
    --model-output models/registry/

python -m sku_mapping.cli evaluate \
    --model models/registry/alkabeer_sku_matcher_v2.joblib \
    --features data/processed/test_features.parquet

python -m sku_mapping.cli run-inference \
    --flyer data/raw/Alkabeer_Export_Data_Clickflyer.csv \
    --master data/raw/Product_Master.xlsx \
    --model models/registry/alkabeer_sku_matcher_v2.joblib \
    --output data/outputs/

Use argparse or Typer.

==================================================
MILESTONE 19 — DOCUMENTATION
==================================================

Write a README explaining:

1. Business objective.
2. Architecture.
3. Training flow.
4. Inference flow.
5. Candidate generation.
6. Feature generation.
7. Model decisions.
8. Threshold policy.
9. Input schemas.
10. Output schemas.
11. Installation.
12. Training command.
13. Inference command.
14. Testing command.
15. Model versioning.
16. How to add a new feature safely.
17. How to retrain after correcting mappings.
18. How to deploy a new model.
19. How to roll back to an older model.
20. Known limitations.

Include architecture diagrams in Mermaid.

Training diagram:

flowchart TD
    A[Gold Pair Dataset] --> B[Validation]
    B --> C[Shared Feature Generator]
    C --> D[Group-Aware Split]
    D --> E[LightGBM Training]
    E --> F[Threshold Tuning]
    F --> G[Evaluation]
    G --> H[Versioned Model Package]

Inference diagram:

flowchart TD
    A[New ClickFlyer Offer] --> B[Preprocessing]
    B --> C[Candidate Generator]
    C --> D[Shared Feature Generator]
    D --> E[LightGBM Predictor]
    E --> F[Decision Policy]
    F --> G{Decision}
    G -->|AUTO_MATCH| H[Accepted SKU]
    G -->|MANUAL_REVIEW| I[Review Queue]
    G -->|NO_MATCH| J[No Match]
    H --> K[Competitor Discovery]

==================================================
IMPLEMENTATION RULES
==================================================

1. Preserve existing working logic first.
2. Refactor in small verifiable stages.
3. Do not combine refactoring and algorithm changes without documenting both.
4. Add tests before materially changing matching behaviour.
5. Avoid unnecessary abstractions.
6. Keep performance suitable for hundreds of thousands of flyer rows.
7. Prefer vectorised pandas and RapidFuzz batch operations.
8. Do not add paid services.
9. Do not add an LLM dependency.
10. Keep the pipeline runnable locally on Windows.
11. Keep compatibility with Kaggle for training.
12. Use UTF-8-SIG for user-facing CSVs where Excel compatibility matters.
13. Use Parquet for internal processed datasets where practical.
14. Use pathlib.
15. Use type hints.
16. Use docstrings.
17. Use explicit exceptions.
18. Do not hide skipped rows.
19. Do not silently change labels.
20. Do not silently infer missing master SKUs.

==================================================
DELIVERY STRATEGY
==================================================

Do not perform the entire refactor blindly in one step.

Deliver it in phases.

For each phase:

1. State which existing functions are being moved.
2. Create the new files.
3. Update imports.
4. Add tests.
5. Run tests.
6. Confirm output parity against the original implementation.
7. Show any intentional behavioural differences.
8. Provide commands to run the phase.

Suggested delivery order:

Phase 1:
- project skeleton
- configuration
- feature generator
- unit tests

Phase 2:
- loaders
- validators
- preprocessing

Phase 3:
- candidate generator
- rule engine
- parity tests

Phase 4:
- training-feature builder
- gold-pair validation

Phase 5:
- LightGBM trainer
- evaluator
- threshold tuner
- model package

Phase 6:
- production predictor
- decision policy
- model compatibility checks

Phase 7:
- competitor discovery
- exports

Phase 8:
- full CLI
- integration tests
- documentation

==================================================
FIRST TASK
==================================================

Start with Phase 1 only.

Inspect the existing sku_mapping_pipeline_ml.py.

Create:

- pyproject.toml
- src/sku_mapping/__init__.py
- src/sku_mapping/constants.py
- src/sku_mapping/config.py
- src/sku_mapping/features/__init__.py
- src/sku_mapping/features/text_features.py
- src/sku_mapping/features/measurement_features.py
- src/sku_mapping/features/semantic_features.py
- src/sku_mapping/features/feature_generator.py
- tests/unit/test_text_features.py
- tests/unit/test_measurement_features.py
- tests/unit/test_feature_generator.py

Extract the feature logic without changing its behaviour.

Expose:

MODEL_FEATURE_COLUMNS

build_feature_vector

build_feature_vector_from_text

Add a parity test that compares the output of the original
_build_ml_feature_row function against the new build_feature_vector
function for representative offer/master pairs.

Do not modify the production pipeline yet beyond optionally importing the
new feature function behind a controlled compatibility flag.

At the end, provide:

1. Created-file list.
2. Explanation of each file.
3. Commands to install dependencies.
4. Commands to run tests.
5. Test results.
6. Any assumptions.
7. Any detected bugs in the original feature logic.
8. A migration plan for Phase 2.

==================================================
PHASE 6A DEPLOYMENT CONTRACT
==================================================

ML deployment is controlled by one explicit mode: `disabled`, `shadow`, or
`assisted`. The repository default is `disabled`. Shadow mode remains
observational and runs only after authoritative exports. Assisted mode applies
the registered v3 package before final mapping and competitor discovery, while
continuing the observational monitoring stream.

The assisted auto-accept threshold is configuration-owned. The initial value
is 0.85 and must be recorded as `threshold_source=user_configured` and
`production_threshold_approved=false`; it does not modify immutable model or
registry metadata. Values below the threshold route to manual review. Hard
semantic, measurement, feature, catalogue, package, and prediction conflicts
block automatic acceptance. Model and monitoring failures are nonfatal and
retain the existing production decision as a safe fallback. Inference never
fits or updates a model.

==================================================
PHASE 6B EMBEDDING SECOND OPINION
==================================================

Embedding scoring is an independent, disabled-by-default observer of the
exact candidate rows retained by RapidFuzz and scored by LightGBM. It cannot
generate, add, filter, or reorder candidates and is not an input to the Phase
6A decision policy.

Backends are configuration-selected and loaded lazily. The preferred backend
is the free local `sentence-transformers/all-MiniLM-L6-v2` model, installed
through the optional `embedding` dependency group. A deterministic local
hashing backend supports offline tests and operational dry runs. No paid API
is supported.

Prepared text retains labelled brand, product/family, variant, offer
description, weight/unit/pack text, master description, category, and Product
Master specification. Persistent vectors are isolated by exact normalized
text, model ID, and model version. Missing or failed embedding infrastructure
must produce an explicit unavailable state and leave LightGBM plus existing
safety policy unchanged.

Phase 6B does not define agreement thresholds, combine model scores, approve
automatic matching, or add an LLM reviewer.

==================================================
PHASE 6C CANDIDATE AGREEMENT AND ROUTING
==================================================

Phase 6C evaluates the independent LightGBM and embedding rankings over the
same retained candidate set. It produces one explicit, offer-level agreement
record without modifying candidate generation, fitting a model, calling an
LLM, or changing the learning dataset.

The default safe-agreement gate requires a valid calibrated LightGBM
probability of at least 0.85, the same top master SKU from both scorers, an
existing master SKU, and no configured hard conflict. A same-candidate result
below the LightGBM threshold is weak agreement and routes to `LLM_REVIEW`.
Different top candidates route to `LLM_REVIEW`. Hard conflicts route to
`MANUAL_REVIEW`. Missing embedding or LightGBM output is recorded explicitly
and uses `SAFE_FALLBACK`; scorer failure is never treated as agreement.

Optional minimum embedding similarity and margin gates remain unset by
default. They may be enabled only through configuration after evidence
supports a threshold. Phase 6C routing remains diagnostic and does not replace
the authoritative production decision.

==================================================
PHASE 6D STRUCTURED LLM REVIEW
==================================================

Phase 6D reviews only offers whose Phase 6C route is `LLM_REVIEW`. It receives
one bounded structured offer plus at most the configured number of retained
candidates. It may accept one supplied candidate, reject all, or return
uncertain. The explicit parser rejects arbitrary SKU IDs and malformed or
out-of-schema output.

The provider boundary supports a lazy local Ollama-compatible implementation
without a paid dependency. LLM review is disabled by default. Timeout,
provider, parsing, and cache failures are nonfatal and route to manual review.
Deterministic protein, family, known measurement, pack, mixed-protein,
feature, and catalogue conflicts always block LLM acceptance.

Response caching is isolated by canonical structured request, exact model ID,
prompt version, and response-schema version. Artifacts contain hashes and
parsed provenance but no endpoint credentials or raw response text.

`LLM_ACCEPT` is diagnostic eligibility only in Phase 6D. No production
decision, Product Master row, model package, model weight, training dataset,
human label, or competitor-discovery input is modified.

==================================================
PHASE 6E UNIFIED ASSISTED INFERENCE
==================================================

Phase 6E composes the existing RapidFuzz candidate generator, shared feature
builder, registered calibrated LightGBM scorer, embedding scorer, agreement
policy, and structured LLM reviewer into one explicit inference orchestrator.
It reuses the shadow scorer chain so every scorer sees the same retained
candidate rows and the existing monitoring/review artifacts continue.

Final decisions are `AUTO_ACCEPT`, `LLM_ACCEPT`, `MANUAL_REVIEW`, `NO_MATCH`,
`NO_CANDIDATE`, `MASTER_SKU_NOT_FOUND`, and `MODEL_ERROR`. Only
`AUTO_ACCEPT` and valid `LLM_ACCEPT` are eligible mappings. Hard deterministic
conflicts, missing embeddings, disabled/failed/uncertain/invalid LLM review,
and unsupported routes are not eligible.

Disabled mode does not call the orchestrator or add output columns. Shadow
mode may calculate the same diagnostic decisions but cannot apply them to
production-owned rows. Assisted mode appends provenance and applies only the
explicit final policy before competitor discovery. Competitor discovery
filters on both final eligibility and the allowed final-decision set.

The legacy master-SKU aggregate exports retain their required columns and
ordering. Assisted mode additionally writes the offer-level
`FINAL_sku_mapping_decisions.csv` with original fields followed by unified
decision and provenance fields.

Phase 6E is assisted inference plus review-data collection. It never fits,
updates, registers, or retrains a model and is not self-learning.

==================================================
PHASE 7A PERSISTENT LEARNING AND REVIEW STORAGE
==================================================

Phase 7A records unified inference runs, every retained candidate prediction,
model/embedding/LLM provenance, deterministic five-question human-review
sessions, automated label trust, model-version observations, and prospective
training-dataset manifests in a versioned repository-owned SQLite store.

Exactly five unique offers are selected after each successful run when at
least five eligible offers exist: high-confidence automatic acceptance, near
the configured threshold, LightGBM/embedding disagreement, LLM-reviewed, and
difficult/conflict-prone. Empty categories use deterministic fallbacks whose
reasons are persisted.

Human review is True/False. False requires a supplied corrected candidate,
none of the supplied candidates, or an explicit cannot-determine response.
Decisive human answers are GOLD. LLM-qualified proposals are SILVER,
model-only labels are PSEUDO and never automatically training-eligible, and
inconclusive/conflicting labels are REJECTED.

Prospective training-dataset records accept GOLD review IDs only and persist
sealed challenge-exclusion proof. Creating these metadata records does not
materialize a training table, fit a model, or modify the registry. Learning
store failures remain nonfatal to Phase 6 production inference.

==================================================
PHASE 7B STREAMLIT REVIEW DASHBOARD
==================================================

Phase 7B exposes upload, processing, five-question review, and validated
download workflows through a local Streamlit presentation layer. Streamlit
pages contain presentation state only and call reusable dashboard services.
Candidate generation, shared features, model scoring, agreement, LLM policy,
competitor discovery, export validation, registry validation, and SQLite
governance remain outside page files.

Uploads are limited by configurable byte size, restricted to CSV/Excel,
signature/schema checked, filename-sanitized, SHA-256 identified, and staged
under generated run directories. Active or completed runs with identical
source bytes are blocked unless the user explicitly confirms reprocessing.
Only registry-listed packages that pass the existing strict package validator
may be selected.

The operational threshold remains 0.85 with
`threshold_source=user_configured` and
`production_threshold_approved=false`. Shadow processing stays observational;
only assisted final decisions explicitly marked eligible may reach competitor
discovery.

SQLite is the durable source for run selection, progress, five questions, and
answers. Browser refresh does not lose completed processing or reviews. False
answers require a supplied correction, none-of-candidates, or
cannot-determine state. No upload or review action invokes training,
calibration, package registration, or model activation.

Downloads use run-safe filenames and are offered only when stored paths are
inside approved roots and CSV/JSON schemas validate. The dashboard never
renders server paths, stack traces, or secret configuration.
## Phase 7C controlled retraining and promotion

Phase 7C is an offline, operator-triggered workflow. Upload inference and
human-review submission never retrain a model.

The workflow materializes immutable baseline-plus-review snapshots, reserves
recent GOLD labels for same-row champion/challenger evaluation, creates
connected-component-safe train/validation/calibration splits, fits weighted
LightGBM challengers, and preserves separate calibration.

PSEUDO labels are excluded. SILVER labels are disabled by default and receive
a lower configurable sample weight when explicitly enabled. Sealed challenge
artifacts remain inaccessible to ordinary training and evaluation.

Challengers are not registered until every conservative comparison check
passes. Passing registration grants assisted-use eligibility only.
Registration never activates a model. Activation is an explicit registry
compare-and-swap with actor, reason, expected champion, history, and rollback.
Automatic production matching remains disabled.
