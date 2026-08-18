# PromoStrater — Developer Handover Guide

> **Audience:** a developer who knows software engineering but nothing about this project.
> **Verified against `main`**, which now contains the `stage1-ml-only-routing` work (merged in PR #1). Older clones and any branch predating that merge still carry the removed embedding architecture; see §15.

---

## 1. What is PromoStrater?

Supermarkets publish promotional flyers. A data vendor ("ClickFlyer") scrapes them into one big spreadsheet: every promoted product, its price, retailer, pack size and brand. Al Kabeer is a frozen-food manufacturer. They want to know two things from that dump.

| | Question | Population |
|---|---|---|
| **Own-SKU mapping** | "This flyer row is one of *our* products — which catalogue item is it?" | Al Kabeer offers |
| **Competitor mapping** | "Which *rival* products in this dump compete with our catalogue item?" | Everything not Al Kabeer |

**Input:** one ClickFlyer CSV/XLSX (the shipped sample is ~234k rows).
**Output:** a SKU mapping table, a competitor relationship table, and an audit table — written to `outputs/dashboard_runs/<run_mode>/<run_id>/` and recorded in SQLite. `<run_mode>` is `production` or `developer`; see §12.

**Example.** The flyer row `Al Kabeer Chicken Samosas 240g` must be recognised as catalogue item `CKSA` ("12 CHICKEN SAMOSAS", 240g). Separately, the rival row `Al Islami Chicken Samosa 2x240gm` must be attached to `CKSA` as a competitor. Neither relationship is given in the source data; both are inferred.

"Master SKU" = a row in `Product_Master.xlsx`, identified by `Itemcode` (e.g. `CKSA`). That is the thing everything gets mapped *to*.

---

## 2. Big picture

Both populations ask the same catalogue the same question with the same machinery. The only thing that differs is the *business relationship* being established.

```
                    ClickFlyer dump (CSV/XLSX)
                              │
                              ▼
                     preprocessing              data/preprocessing.py
                     · normalise text, parse pack sizes
                     · categorise, derive product_family
                     · set is_own  (brand == Al Kabeer?)
                              │
                     ┌────────┴────────┐
                     │  is_own split   │
                     └────┬───────┬────┘
              is_own=True │       │ is_own=False
                          ▼       ▼
                    OWN-SKU     COMPETITOR
                          │       │
                          └───┬───┘
                              ▼
                  SAME candidate generator      matching/candidate_generator.py
                  (shortlist of Master SKUs)
                              ▼
                  SAME 41 features              features/
                              ▼
                  SAME LightGBM                 ranked-v5-cal package
                              ▼
                  SAME routing                  matching/routing.py
                  ┌───────────┴───────────┐
              LLM ON                   LLM OFF
              ≥0.95 AUTO               ≥0.85 AUTO
              <0.95 → Gemini           <0.85 → Human Validation
                  └───────────┬───────────┘
                              ▼
                    final relationships
                    · own      → "this offer IS SKU X"
                    · competitor → "this rival offer COMPETES WITH SKU X"
                              ▼
                  exports + SQLite + dashboard downloads
```

> **Competitors are hybrid, by measurement.** Own-brand runs the flow above.
> Competitors deliberately do **not** use the model as their classifier:
> measured on a real samosa slice, replacing the rules with the own-brand
> model cut CKSA from **194 competitors to 4** and lost Biladi entirely
> (0.2477 against a 0.85 cut). The package is trained on Al Kabeer→Al Kabeer
> pairs and scores rival text out of domain, so it ranks — it never admits or
> rejects.
>
> The live competitor path is **retrieval → ML ranking → automatic policy →
> Gemini**, all inside `competitors/discovery.py`. Recall is measured as
> preserved: 307/307 relationships, zero lost. See §7.

---

## 3. Repository map

| Folder | What it holds | Why it exists |
|---|---|---|
| `src/sku_mapping/` | All domain logic. Importable, no Streamlit. | The engine. Everything else is a caller. |
| `src/sku_mapping/data/` | Loaders, validators, preprocessing, offer identity. | Turns a raw dump into normalised rows with `is_own`, `category`, `match_text`, pack measures. |
| `src/sku_mapping/matching/` | Candidate generation, rules, **routing**, shared matcher. | Shortlisting and the single threshold authority. |
| `src/sku_mapping/features/` | Feature builders (text, measurement, semantic, rank, discriminative). | Produces the numeric vector the model consumes. |
| `src/sku_mapping/ml/` | Model package loading, calibration, predictors, safety checks. | Loads and validates the registered model; refuses mismatched ones. |
| `src/sku_mapping/inference/` | `run_unified_inference` — the orchestrator. | The one entry point that runs a whole upload. |
| `src/sku_mapping/competitors/` | Discovery, policy, adjudicator, decisions, re-ranker. | The competitor half of the product. |
| `src/sku_mapping/llm_review/` | Provider boundary, Gemini, parser, cache. | Second-stage reviewer. Never a matcher. |
| `src/sku_mapping/learning/` | SQLite store, migrations, observers, review selection. | Durable record of runs, predictions, decisions, human answers. |
| `src/sku_mapping/incremental/` | Cumulative offer state and delta planning. | Lets a weekly load infer only what is new. See §12. |
| `src/sku_mapping/retraining/` | Snapshots, challenger training, comparison, activation. | Offline champion/challenger. Never runs during an upload. |
| `src/sku_mapping/shadow/` | Observational scoring pipeline. | Scores without affecting production rows. |
| `dashboard/` | Streamlit UI + application services. | The only user interface. |
| `config/` | `default.yaml`. | Every runtime knob. |
| `models/` | Registered model package + registry JSON. | The trained artefact; required to run. |
| `tests/` | 63 unit + 16 integration files. | Encodes the safety boundaries. |
| `docs/` | Architecture, policies, this guide. | Written rationale. |
| `scripts/` | ~33 offline audit/evaluation/training scripts. | Never part of an upload run. |

---

## 4. Important files

| File | Purpose | Why you care |
|---|---|---|
| `src/sku_mapping/inference/pipeline.py` | `run_unified_inference()` — orchestrates one upload end to end. | Start here to follow any run. |
| `src/sku_mapping/matching/routing.py` | Thresholds + review destination. | **The only place a threshold is chosen.** |
| `src/sku_mapping/matching/candidate_generator.py` | Builds the shortlist of Master SKUs per offer. | If the right SKU never appears, the model can't pick it. |
| `src/sku_mapping/constants.py` | `MODEL_FEATURE_COLUMNS` (19 base features). | The shared training/inference contract. |
| `src/sku_mapping/features/feature_generator.py` | Assembles the per-pair feature vector. | Where a new feature goes. |
| `src/sku_mapping/ml/model_package.py` | Loads and validates the model package. | Rejects wrong Python/LightGBM/sklearn — a common setup failure. |
| `src/sku_mapping/competitors/discovery.py` | Competitor search over the dump. | The **live** competitor engine today (see §7). |
| `src/sku_mapping/matching/shared_matcher.py` | One engine for own + competitor. | The **target** design; currently unreferenced. |
| `src/sku_mapping/competitors/policy.py` | Automatic ACCEPT/REJECT rules. | Provisional thresholds live here. |
| `src/sku_mapping/competitors/adjudicator.py` | LLM decision for ambiguous competitors. | Last stage before a terminal decision. |
| `src/sku_mapping/llm_review/gemini.py` | Gemini HTTP provider. | Prompt→text only; no policy. |
| `src/sku_mapping/learning/store.py` | All SQLite access. | Every read/write to the database. |
| `src/sku_mapping/learning/migrations.py` | Schema versions. | Any column change starts here. |
| `src/sku_mapping/incremental/state.py` | Delta planning + cumulative frames. | Decides what a weekly load actually reprocesses. |
| `dashboard/services/processing_service.py` | Runs a job: validate → inference → exports → persist. | The bridge from UI to engine. |
| `config/default.yaml` | Typed configuration. | Toggles for Gemini, competitors, thresholds. |

---

## 5. What happens to one upload

| # | Stage | Where | What it does |
|---|---|---|---|
| 1 | Upload & validate | `dashboard/services/upload_service.py` | Checks extension/size, hashes the bytes, refuses an exact duplicate of a previous run **in the same run mode**. Developer mode never refuses a repeat. |
| 2 | Job created | `dashboard/services/job_manager.py` | Writes a `processing_jobs` row and starts a daemon worker thread. State lives in SQLite so a refresh or second tab shows the same run. |
| 3 | Preprocess | `data/preprocessing.py` | Normalises text, parses pack measures, derives `category` and `product_family`, sets `is_own` by brand. |
| 4 | Offer identity | `data/offer_identity.py` | Assigns a stable `offer_group_id` (the vendor `offerid`, or a deterministic fingerprint). Rows collapse to one per identity for inference. |
| 4b | Delta planning (production only) | `incremental/state.py` | Selects the own-brand offers that are new, content-revised, or previously unmapped against a changed Product Master. Everything else skips inference. Developer mode skips this step entirely. |
| 5 | Entity expansion | `inference/pipeline.py` | Splits a multi-product offer ("Nuggets + Samosa") into separate entities so each can map to its own SKU. |
| 6 | Candidates | `matching/candidate_generator.py` | Fuzzy shortlist of plausible Master SKUs per offer. |
| 7 | Features | `features/feature_generator.py` + rank/discriminative builders | Builds the 41-column numeric frame for every (offer, candidate) pair. |
| 8 | Score | `ml/ranked_predictor.py` | The registered LightGBM package scores each pair. |
| 9 | Route | `matching/routing.py` | Applies the active threshold; sends the residue to Gemini or to the human queue. |
| 10 | Competitor relationships | `competitors/discovery.py` (via `exports/business_outputs.py`) | Establishes which rival offers compete with which Master SKU. See the note below. |
| 11 | Export | `exports/business_outputs.py`, `exports/run_outputs.py` | Writes the SKU mapping, competitor aggregate, and long-form audit — always over the **cumulative** offer set, never just this file's delta. See §12. |
| 12 | Persist | `learning/observer.py` → `learning/store.py` | Records the run, predictions, and decisions. Five review questions are selected if eligible. |

**How the two populations relate.** Preprocessing sets `is_own` once (step 3), and that flag is the only thing that separates them. Steps 6–9 — candidates, features, model, routing — are the shared engine, and the global LLM toggle governs both. Competitor results are grouped back by Master SKU at export time, because a competitor answer is *reported* per SKU even though it is *decided* per offer.

> **Sequencing, accurately.** Steps 6–9 currently run for the **own-brand**
> population. Step 10 is then invoked from `build_business_outputs`, and today
> it uses its own rule-gate + fuzzy engine rather than steps 6–9. So the
> present code does run competitors after own-brand mapping — not because the
> design requires it, but because the competitor path has not yet been moved
> onto the shared matcher. Don't read the current ordering as intentional
> architecture.

Inference **never trains**. No stage in this list fits a model.

---

## 6. Own-SKU mapping

**Master SKU catalogue** — `Product_Master.xlsx`, keyed by `Itemcode`. Preprocessing derives a `category` from `Item-Cat-2` and a normalised family from `Item-Cat-4`.

**Candidate generation** — makes a shortlist of possible SKUs using fuzzy text similarity. Cheap, recall-oriented. Nothing downstream can recover a SKU that never made the shortlist.

**The 41 features** — built per (offer, candidate) pair:

| Block | Count | Examples |
|---|---|---|
| Base (`MODEL_FEATURE_COLUMNS` in `constants.py`) | 19 | `protein_match`, `family_match`, `size_match`, `word_similarity`, `total_offer_weight_g` |
| Discriminative | 6 | `size_ratio`, `spice_conflict`, `product_line_conflict`, `bulk_mismatch` |
| Rank-aware | 16 | `word_similarity__rank`, `__minus_max`, `__is_best`, `__z` for 4 base features |

Rank features describe a candidate *relative to its own shortlist* — "is this the best match for this offer?" — which is what makes the model a ranker rather than a per-pair classifier.

> **Gotcha:** `MODEL_FEATURE_COLUMNS` has **19** entries, but the registered model records **41**. Both are correct: 19 is the base block, 41 is what the model consumes. Don't "fix" one to match the other.

**LightGBM** — the registered package `ranked-v5-cal-20260810T100756Z-matcher`. Loading validates feature order, calibration state, and the runtime Python/LightGBM/scikit-learn versions; a mismatch raises rather than silently degrading.

**Threshold routing** — `matching/routing.py` owns this and nothing else does:

| Global toggle | Auto-accept at | Below goes to |
|---|---|---|
| `llm_review.enabled: true` | **0.95** | Gemini (run finishes with no human) |
| `llm_review.enabled: false` | **0.85** | Human Validation queue |

Two different numbers on purpose: with a reviewer behind it the cut can be strict, because being sent to Gemini is cheap; with no reviewer the same strictness would bury a person, so it relaxes and the residue becomes an explicit queue.

> **These are MODEL SCORE cut-offs, not accuracy.** `0.95` does not mean "95% correct". Nothing in this repository has measured accuracy at either point. Never label them "confidence" on a user-facing surface.

Flipping the toggle changes *only* the threshold and the destination. Candidates, features, model and scores are identical in both modes — asserted in `tests/unit/test_toggle_production_wiring.py`.

---

## 7. Competitor mapping

### The conceptual difference

Both populations map an offer to a Master SKU. What differs is the *claim* being made:

| | Asks | Claim |
|---|---|---|
| **Own-SKU** | "Which Master SKU **is** this offer?" | identity — this row *is* our product |
| **Competitor** | "Which Master SKU does this rival offer **map to** for competitive purposes?" | rivalry — this row *competes with* our product |

The matching machinery is meant to be identical; only the interpretation changes. A competitor never needs to be mapped first — the relationship is established directly.

### Target architecture

```
non-Al-Kabeer offer
        ▼
shared candidate generator
        ▼
same 41 features
        ▼
same LightGBM
        ▼
global routing (same toggle, same thresholds)
        ▼
competitor → Master SKU relationship
```

`matching/shared_matcher.py` implements exactly this. It is **not** the competitor production path, and deliberately so: applying the own-brand model as a competitor classifier was measured to cut CKSA from 194 competitors to 4 and lose Biladi. It remains as the reference design for the day a competitor-trained model exists.

### Live implementation

Today the relationship is established by `competitors/discovery.py`, reached from `build_business_outputs`. It iterates each mapped Master SKU against every non-own offer and applies rule gates before a fuzzy score:

| Gate | Rejects with |
|---|---|
| Same `category`? | `CATEGORY_CONFLICT` |
| Same `product_family` **or** semantic family overlap? | `PRODUCT_FAMILY_CONFLICT` |
| Protein sets compatible? | `PROTEIN_CONFLICT` (hard) |
| Family sets compatible? | `FAMILY_CONFLICT` (hard) |
| Pack size within ±10%? | `PACK_CONFLICT` (hard) |
| Fuzzy score ≥ `raw_score_floor` (60) **and** adjusted ≥ `adjusted_score_floor` (65) | `BELOW_COMPETITOR_ELIGIBILITY_POLICY` |

Survivors become `MATCHED` (pack verified) or `AMBIGUOUS` (pack unknown). Every rejection carries a reason code, so the wide aggregate is provably a projection of the long-form audit table.

On top of that sit the layers that move competitors toward fully automatic decisions. **All are off by default**, so a stock checkout runs rules-only:

| Config flag | Module | Effect |
|---|---|---|
| `ml_reranking_enabled` | `competitors/reranker.py` | Reorders the surviving shortlist with the own-brand LightGBM package. **Ranking only** — cannot admit or reject a candidate, cannot override a conflict, and its output is a raw margin, never a probability. |
| `automatic_decisions_enabled` | `competitors/policy.py` → `decisions.py` | Turns every relationship into a terminal ACCEPT/REJECT using margin and gap thresholds. No human is involved. |
| `llm_adjudication_enabled` | `competitors/adjudicator.py` | Sends the cases policy cannot settle to Gemini. Anything unresolved **rejects**. |
| `review_staging_per_target` | `competitors/review.py` | **Offline only.** Stages top-ranked competitors for human labelling to build ground truth. Not part of the production decision path. |

**There is no production human route for competitors.** With the decision layers enabled, every relationship ends ACCEPTED or REJECTED automatically. Human review of competitors exists solely as a development/evaluation mechanism (page 6 + `review_staging_per_target`, default `0`).

### Module status — verified by actual callers

| Module | Status | Evidence |
|---|---|---|
| `competitors/discovery.py` | **ACTIVE** | Called from `exports/business_outputs.py:404`, reached from `processing_service.py:571`. |
| `competitors/text_normalisation.py` | **ACTIVE** | Imported by `discovery.py` and `reranker.py`. |
| `competitors/policy.py` | **ACTIVE** | `DECISION_COLUMNS` imported by `discovery.py`; thresholds used by `decisions.py`. |
| `competitors/decisions.py` | **ACTIVE (gated)** | Imported lazily at `discovery.py:1293`; runs when `automatic_decisions_enabled`. |
| `competitors/adjudicator.py` | **ACTIVE (gated)** | Imported by `decisions.py`; runs when `llm_adjudication_enabled`. |
| `competitors/reranker.py` | **ACTIVE (gated)** | `load_competitor_reranker` imported by `processing_service.py:198`. |
| `competitors/review.py` | **ACTIVE (offline)** | Imported by `dashboard/pages/6_Competitor_Review.py`. |
| `matching/shared_matcher.py` | **UNUSED (by decision)** | Zero importers. Pure-ML competitor matching was measured to destroy recall, so the hybrid path is production instead. |
| `agreement/` + `agreement:` config block | **TRANSITIONAL** | Survives from the older two-scorer design. `routing.py` is the threshold authority; `agreement.lightgbm_auto_accept_threshold` overlaps confusingly. |
| `embedding/` | **REMOVED** | Package deleted. References in old docs or on `main` are stale. |

None of these is dead code you can delete — the gated ones run as soon as their flag is set. The one genuinely unreferenced module is `shared_matcher.py`, and it is unreferenced because it is ahead of the wiring, not behind it.

---

## 8. Gemini

| | |
|---|---|
| **Provider** | `llm_review/gemini.py` — turns a prompt into response text. Nothing else. |
| **Boundary** | `llm_review/provider.py` — the swappable interface. A local Ollama implementation also lives here. |
| **Policy/parsing** | `llm_review/reviewer.py` — schema validation, candidate-id checking, confidence policy, retries, caching. |
| **When called** | Only for offers below the auto-accept threshold, and only when `llm_review.enabled: true`. |
| **What it receives** | The offer text plus a bounded list of candidates the pipeline already produced (`maximum_candidates: 5`). |

**It cannot invent a SKU.** The reviewer validates every returned candidate id against the list it supplied; anything unrecognised is discarded. Gemini is a *reviewer*, never the matcher.

### Failure behaviour differs by population

The two populations have different safe directions, and mixing them up is the easiest mistake to make here.

**Own-SKU** — the residue can land on a person:

| Toggle | Below threshold goes to | If Gemini fails |
|---|---|---|
| LLM **ON** | Gemini | `fail_route: manual_review` |
| LLM **OFF** | Human Validation directly | no API call is made at all |

**Competitor** — the residue must never need a person:

| Gemini outcome | Result |
|---|---|
| ACCEPT (valid supplied candidate) | automatic competitor match |
| REJECT | automatic rejection |
| UNCERTAIN, malformed reply, timeout, API failure, invented/unrecognised SKU | **conservative automatic rejection** |

Every competitor failure mode converges on REJECTED. That is deliberate: a missed competitor understates a rival's presence, while a wrong one asserts a rivalry that does not exist — so rejection is the safe direction. **Competitor production has no human-review dependency.**

> **Note:** `config/default.yaml` currently sets `llm_review.provider: ollama`, not `gemini`, and `enabled: false`. Gemini is implemented but not the configured default. Its API key is read from the environment, never from config.

---

## 9. Database

**Location:** `data/learning/sku_learning.db` (SQLite, WAL mode). Path from `learning_store.database_path`.

**All access** goes through `learning/store.py`. Nothing else opens the file.

| Table | Holds |
|---|---|
| `pipeline_runs` | One row per upload run. Carries `run_mode` (`production` / `developer`). |
| `processing_jobs` | Worker job state — what the dashboard polls for live status. |
| `predictions` | Candidate-level observations (scores, decisions) per run. |
| `offer_decisions` | The final per-offer outcome. |
| `competitor_decisions` | Automatic competitor ACCEPT/REJECT records. |
| `review_sessions`, `human_reviews` | The five-question review flow and its answers. |
| `automated_labels` | Machine-generated labels with provenance. |
| `model_versions`, `training_datasets` | Registered model and dataset lineage. |
| `offer_ledger` | One row per offer ever processed: content hash, master hash, mapped flag. Drives incremental loading. |
| `schema_migrations` | Applied schema version. |

**Migrations** live in `learning/migrations.py` and run automatically on connect. Adding or changing a column means adding a migration — never editing the table definition in place. The current version is **10** (`run_mode` + `offer_ledger`).

`offer_ledger` deliberately has **no foreign key** to `pipeline_runs`. It is cumulative state that must outlive any individual run record: a cascade from a deleted run would silently destroy the incremental history and send the next weekly load back to a full reprocess.

Readers and writers are deliberately separated: bulk inserts hold a write lock, while job-status reads take a lock-free path (WAL gives them a consistent snapshot). A shared lock previously froze the dashboard's live panel for whole stages.

---

## 10. Dashboard

```
Streamlit pages  (render only)
        ▼
dashboard/services/*   (validate, orchestrate, persist)
        ▼
src/sku_mapping/inference/pipeline.py
        ▼
SQLite + outputs/dashboard_runs/<run_mode>/<run_id>/
        ▼
Results pages read back from the store
```

Pages hold no business logic; domain modules never import Streamlit.

| Page | Purpose |
|---|---|
| `1_Upload_and_Process.py` | Upload, start a run, live progress. Hosts the **Developer mode** toggle. |
| `2_Human_Validation.py` | The five-question review flow. |
| `3_Results_and_Downloads.py` | Run outcome and file downloads. |
| `4_Models_and_Learning.py` | Registered model info and learning-store stats. |
| `5_Al_Kabeer_SKUs.py` | Per-SKU view: mapped offers and competitors. |
| `6_Competitor_Review.py` | Competitor labelling surface. |

| Service | Purpose |
|---|---|
| `processing_service.py` | Runs a job end to end. **The bridge from UI to engine.** |
| `job_manager.py` | Worker threads, job lifecycle, orphan recovery. |
| `pipeline_status.py` | One status snapshot both the page and sidebar read, so they can't disagree. |
| `upload_service.py` | Validation, hashing, duplicate prevention (scoped per run mode). |
| `run_service.py` / `review_service.py` | Run history; review answers. |
| `registry_service.py` / `model_insights_service.py` | Model listing and plain-language model info. |

Processing runs on a **daemon thread**, with state in SQLite rather than session state — so a browser refresh or a second tab shows the same live run.

---

## 11. Configuration

`config/default.yaml`. Paths resolve relative to that file.

| Setting | Default | Meaning |
|---|---|---|
| `llm_review.enabled` | `false` | **The global toggle.** Selects the threshold *and* the review destination for both populations. |
| `llm_review.on_auto_accept_threshold` | `0.95` | Auto-accept cut when the reviewer is on. |
| `llm_review.off_auto_accept_threshold` | `0.85` | Auto-accept cut when it's off. |
| `llm_review.provider` | `ollama` | Provider selection. Gemini exists but isn't the default. |
| `ml.model_id` | `ranked-v5-cal-…` | Which registered package to load. |
| `ml.require_registered_model` | `true` | Refuse to score without a valid registered package. |
| `ml.mode` | `assisted` | `disabled` / `shadow` / `assisted`. |
| `shadow_mode.chunk_size` | `10000` | Offers per streaming chunk. Execution only — results are identical. |
| `competitors.raw_score_floor` / `adjusted_score_floor` | `60` / `65` | Fuzzy eligibility floors. |
| `competitors.max_per_target` | `0` | `0` = no limit; keep every competitor clearing the floors. |
| `competitors.ml_reranking_enabled` | `false` | Order survivors with the model. |
| `competitors.automatic_decisions_enabled` | `false` | Terminal ACCEPT/REJECT without a human. |
| `competitors.llm_adjudication_enabled` | `false` | Ask the LLM about ambiguous competitors. |

### The one global toggle

`llm_review.enabled` is a single switch governing **both** populations:

| | Auto-accept at | Below goes to |
|---|---|---|
| **ON** | `0.95` | Gemini — the run finishes with no human |
| **OFF** | `0.85` | Human Validation queue (own-SKU); no API call is made |

What the toggle changes: **the threshold and the review destination. Nothing else.**

| | Changes with the toggle? |
|---|---|
| Candidate generation | No — identical |
| The 41 features | No — identical |
| LightGBM model and scores | No — identical |
| Threshold + review destination | **Yes** |

That equivalence is asserted in `tests/unit/test_toggle_production_wiring.py`, and it is the reason the toggle is safe to flip on a live deployment.

Run mode is **not** configured here. It is a per-run choice made on the Upload page and recorded on the run itself (§12), because two runs started minutes apart may legitimately want different answers.

Never put credentials in this file. The Gemini key comes from the environment.

---

## 12. Run modes and incremental loading

Two features, one idea: a run declares **which body of state it belongs to**,
and that declaration decides what it reads, what it writes, and who sees it.

### The toggle

`Developer mode` on the Upload page. Both modes execute the **identical**
pipeline — candidate generation, scoring, Gemini review, competitor
discovery, review staging. Nothing is stubbed or skipped in developer mode.
What differs is state.

| | Production | Developer |
|---|---|---|
| Outputs | `outputs/dashboard_runs/production/` | `outputs/dashboard_runs/developer/` |
| Offer ledger | reads it, then advances it | never touched |
| Same file uploaded twice | refused as a duplicate | always runs, always in full |
| Run pickers (`2_Human_Validation`, `3_Results_and_Downloads`) | shown | shown — they follow the active toggle, so a developer can see their own results |
| `5_Al_Kabeer_SKUs` (the business view) | shown | **never** shown — pinned to production |
| Training snapshot + retrain gate | included | excluded by default |

Runs were **always** isolated by `run_id`, so nothing ever overwrote anything
and the directory split is not what makes developer mode safe. Ledger
participation is. The folders exist so a person browsing the filesystem
cannot mistake an experiment for the week's business output.

**Developer runs still stage human reviews**, deliberately — that is what
makes a developer run a real rehearsal rather than a partial one. What keeps
them out of training is the *selection*, not a blocked code path:
`governed_training_labels()` and `count_new_gold_labels_since_last_model()`
both default to `run_mode="production"`. Pass `run_mode=None` to include
every mode, deliberately.

`store.list_pipeline_runs()` defaults to production too, so a caller that
never considered run modes returns an empty list rather than silently
surfacing an experiment. The one deliberate exception is
`pipeline_status.py`, which passes `run_mode=None`: the sidebar must report
the run that is actually happening, whichever mode owns it.

### Incremental loading

A production run infers an own-brand offer only if it is:

1. **new** — never seen before;
2. **revised** — its content hash changed under an unchanged `offerid`, which
   is what a corrected price looks like; or
3. **re-mappable** — it was seen, never matched, and `Product_Master.xlsx`
   has changed since.

Everything else skips inference entirely.

**The watermark is offer identity, never a date.** ClickFlyer dumps carry
backdated rows and corrections, so a "process everything after date X" filter
drops exactly the rows a correction was meant to deliver — silently and
permanently. The latest `Offer End Date` *is* recorded and reported as
`data_through_offer_end_date`, but only so an operator can be told how
current the data is. Nothing selects work by it.
`test_backdated_rows_are_not_dropped` fails if anyone reintroduces that.

**Inference is incremental; outputs are cumulative.** This is the part that
is easy to get wrong. Competitor discovery answers "which rivals compete with
this Master SKU" — a question about *every offer ever seen*, not about this
week's file. Exporting a delta would produce a competitor list that looks
complete and is not. So each run reassembles the full picture from stored
state plus its own delta, and the export layer never learns that incremental
loading exists.

`test_two_incremental_loads_equal_one_full_load` splits a dump in half and
asserts the two-run result is frame-for-frame identical to one full run.
**That is the property to protect.** It is what would break silently if
someone later tries to make the exports incremental too.

### Where the state lives

`outputs/dashboard_runs/_incremental/` — three frames, alongside the existing
`_pipeline` and `_shadow` trees:

| File | Holds |
|---|---|
| `canonical_offers.pkl` | Variant-level pool of every offer ever seen (own + competitor). Feeds competitor discovery. |
| `own_offer_rows.pkl` | One canonical row per offer identity. |
| `decisions.pkl` | Cumulative per-offer decisions. |

Derived from `dashboard.output_directory` rather than configured separately,
so anything that redirects outputs — a test, a second checkout — redirects
cumulative state with it instead of sharing one global history. There is no
YAML key for it, by design.

**Pickle, not Parquet.** Prepared offers carry non-scalar derived features
(`offer_measures_detailed` holds nested measure objects) that Parquet cannot
encode without a lossy type round-trip — and that round-trip would alter the
very values the content hashes are computed over, so every unchanged offer
would read as revised and incremental loading would quietly degrade into full
reprocessing. These are local, single-writer files, never an interchange
format.

**To force a full reprocess:** `store.reset_offer_ledger()` and
`IncrementalStateStore(...).clear()`. Both must be done together — the ledger
decides what is skipped, the frames supply what is carried forward.

### What this does and does not save

Saved: the expensive per-offer work — candidate generation, feature building,
LightGBM scoring, Gemini calls — for every offer already current.

**Not saved:** competitor discovery, which still runs across the full history
every week. That is the price of the export staying complete, and it is the
dominant cost on a large ledger. Also not saved: the first production run,
which processes everything and builds the ledger from scratch.

---

## 13. Tests

63 unit files, 16 integration files. Run everything with `pytest` from the repo root.

| Category | Covers |
|---|---|
| Feature/candidate units | Feature builders, candidate generation, families, measurements. |
| Policy units | Routing modes, toggle wiring, agreement policy, safety thresholds. |
| Competitor units | Discovery gates, decision policy, adjudication, business outputs. |
| Store units | Schema, migrations, reviews, persistence visibility. |
| Run modes + incremental | Mode isolation, ledger behaviour, and the incremental/full equivalence property. |
| Dashboard units | Services, progress, formatters, upload validation. |
| Integration | Whole-pipeline runs, phase boundaries, isolation guarantees, retraining, cancellation. |

**If you change X → run Y**

| Change | Run |
|---|---|
| Features or constants | `tests/unit/test_feature_generator.py`, `tests/integration/test_build_training_features.py` |
| Candidate generation | `tests/unit/test_candidate_generator.py`, `tests/integration/test_candidate_generation_parity.py` |
| Thresholds / routing | `tests/unit/test_routing_mode.py`, `tests/unit/test_toggle_production_wiring.py` |
| Competitors | `tests/unit/test_business_outputs.py`, `tests/unit/test_business_safety_adversarial.py` |
| LLM review | `tests/unit/test_llm_reviewer.py`, `tests/integration/test_llm_review_phase_boundaries.py` |
| Store / migrations | `tests/unit/test_learning_store_schema.py`, `tests/integration/test_learning_store_observation.py` |
| Run modes, the ledger, incremental loading | `tests/integration/test_incremental_and_run_modes.py` |
| Anything touching what gets reprocessed | `tests/integration/test_incremental_and_run_modes.py::test_two_incremental_loads_equal_one_full_load` — **the equivalence property**; if this fails, incremental loading is shipping a different answer than a full run |
| Dashboard | `tests/unit/test_dashboard_services.py`, `tests/integration/test_dashboard_processing_service.py` |
| Model loading | `tests/unit/test_model_package.py`, `tests/unit/test_controlled_model_registry.py` |
| Anything in the orchestrator | `tests/integration/test_unified_inference_pipeline.py` |

---

## 14. Where do I go if…

| I need to change… | Start here |
|---|---|
| Text cleaning, `is_own`, categories, pack parsing | `src/sku_mapping/data/preprocessing.py` |
| Which SKUs get shortlisted (both populations) | `src/sku_mapping/matching/candidate_generator.py` |
| The shared own+competitor engine (reference only) | `src/sku_mapping/matching/shared_matcher.py` — **not production**; needs a competitor-trained model first |
| A feature the model sees | `src/sku_mapping/features/feature_generator.py` + `constants.py` |
| Model loading / validation | `src/sku_mapping/ml/model_package.py` |
| A threshold or where reviews go | `src/sku_mapping/matching/routing.py` — **nowhere else** |
| Gemini prompt, parsing, or failure behaviour | `src/sku_mapping/llm_review/reviewer.py` (policy), `gemini.py` (transport) |
| Competitor gates or scoring (**live path**) | `src/sku_mapping/competitors/discovery.py` |
| Competitor accept/reject rules | `src/sku_mapping/competitors/policy.py` → `decisions.py` |
| Competitor LLM adjudication | `src/sku_mapping/competitors/adjudicator.py` |
| Competitor ranking with the model | `src/sku_mapping/competitors/reranker.py` |
| Anything touching the database | `src/sku_mapping/learning/store.py` |
| A schema/column change | `src/sku_mapping/learning/migrations.py` (add a migration) |
| What a weekly load reprocesses | `src/sku_mapping/incremental/state.py` |
| Which runs a page or service can see | the `run_mode` argument on `store.list_pipeline_runs()` — production by default |
| Forcing a full reprocess | `store.reset_offer_ledger()` **and** `IncrementalStateStore(...).clear()` — both, or the two disagree |
| A page, control, or progress display | `dashboard/pages/`, `dashboard/components/` |
| How a run is orchestrated | `dashboard/services/processing_service.py` |
| Output columns or file layout | `src/sku_mapping/exports/business_outputs.py` |
| Training a new model | `scripts/train_ranked_v5_calibrated.py`, `src/sku_mapping/training/` |
| Promotion / rollback | `src/sku_mapping/retraining/` |
| A runtime knob | `config/default.yaml` |

---

## 15. Current limitations

- **Older clones are far behind.** `main` now contains the `stage1-ml-only-routing` work, but any clone or branch predating that merge still carries the removed embedding architecture and the older two-scorer agreement policy. Check that you have PR #1 before reading the code to learn the system.
- **Thresholds are provisional and unvalidated.** `0.95` / `0.85` are operational choices, never measured against a human-labelled set. The competitor margin/gap thresholds come from an 8,000-row slice with **no human competitor labels at all**.
- **Competitor precision and recall have never been independently validated.** There is no competitor ground truth at all. `review_staging_per_target` exists to start collecting it and defaults to `0`.
- **The competitor migration is mid-flight.** The shared engine is the agreed target and is implemented in `matching/shared_matcher.py`, but nothing imports it yet, so competitors still run the rule-gate + fuzzy path in `discovery.py`. Both are real; neither is dead. Treat §7's status table as the source of truth.
- **The competitor re-ranker borrows the own-brand model**, trained on Al Kabeer→Al Kabeer pairs. It is restricted to ranking for exactly that reason, and its output is an uncalibrated margin.
- **The automatic competitor decision layers are off by default** (`automatic_decisions_enabled`, `llm_adjudication_enabled`, `ml_reranking_enabled` all `false`), so a stock checkout produces rules-only competitor output.
- **`agreement/` overlaps `routing.py`**, leaving two places that look like threshold authorities. Only `routing.py` is.
- **Gemini connectivity is unverified in this repository.** The provider is implemented, but `llm_review.provider` is `ollama` and `llm_review.enabled` is `false`, so no LLM call is made by default and no test exercises a real external API.
- **Pack size is a hard conflict at ±10%**, so a 500g rival pack is never a competitor to a 240g SKU. That's a deliberate business rule, not a bug — but it's the single largest source of competitor rejections after family conflict.
- **Incremental loading does not shorten competitor discovery.** Only per-offer inference is skipped. Competitor discovery is cross-sectional and still runs over the entire ledger every week, so the weekly run does not get faster indefinitely — it converges on the cost of discovery over the full history.
- **The offer content hash is 64-bit** (`hash_pandas_object`). Across a million offers the chance of any collision is roughly three in a hundred million, and the only consequence is that one revised offer is not re-inferred. Cheap enough to accept; worth knowing it is not cryptographic.
- **Cumulative state is pickle.** It is local single-writer state under the gitignored `outputs/` tree, but it is therefore tied to the pandas version that wrote it. A major pandas upgrade may require clearing and rebuilding it (see §12).
- **`mapped` in the ledger is read from decisions, not from the export.** If a future change moves where the final SKU is recorded, `_ledger_records` in `processing_service.py` must move with it — otherwise offers get marked mapped when they are not and are never reconsidered.
- **SQLite, single host.** No multi-user authorisation; the dashboard ships no authentication.

---

## 16. Ten-minute handover talking points

1. **What it does** — maps flyer offers to Al Kabeer's catalogue, and finds which rival offers compete with each catalogue item.
2. **One engine, two claims** — the dump is split by `is_own`, and both populations use the same candidate generator, the same 41 features and the same LightGBM. What differs is the business relationship being established: own-SKU asserts *identity* ("this offer **is** SKU X"), competitor asserts *rivalry* ("this rival offer **competes with** SKU X").
3. **The flow** — preprocess → split by `is_own` → shared candidates → 41 features → LightGBM → routing → final relationships → exports + SQLite.
4. **41 = 19 + 6 + 16** — base contract, discriminative features, rank-aware features. `MODEL_FEATURE_COLUMNS` is only the 19.
5. **One toggle, two thresholds** — `llm_review.enabled` picks 0.95→Gemini or 0.85→Human. Everything else is identical, and tests assert that.
6. **Thresholds are model scores, not accuracy** — never call them confidence.
7. **Gemini reviews, it never matches** — it only chooses among supplied candidates, and the reviewer rejects any id it didn't supply.
8. **Failure directions differ by population** — an own-SKU failure falls back to a human; a competitor failure (uncertain, timeout, malformed, invented SKU) is a conservative automatic REJECT. Competitor production needs no human.
9. **The competitor migration is mid-flight** — the target is the shared matcher; the live path is still `discovery.py`'s rule gates plus fuzzy floors, with the model as an optional re-ranker. Know which one you're editing.
10. **Every competitor rejection has a reason code** — the user-facing aggregate is a projection of the long-form audit, so the two can't disagree.
11. **The database is the source of truth** — job state lives in SQLite, not session state, which is why a refresh doesn't lose a run.
12. **Inference never trains** — retraining is a separate, operator-triggered champion/challenger workflow.
13. **Two run modes, one pipeline** — developer runs execute everything production does, including review staging; they differ only in which state they touch and who can see them.
14. **Weekly loads are incremental, exports are cumulative** — only new or revised offers are inferred, but competitor discovery is a question about every offer ever seen, so the outputs are always rebuilt over the full history. The equivalence test is what holds that line.
15. **Identity decides the work, never a date** — dumps arrive backdated and corrected, so a date watermark drops exactly the rows a correction was meant to deliver.
16. **When something breaks** — read `inference/pipeline.py` for the orchestration, check `processing_jobs` for the run's real state, and remember that `routing.py` is the only place a threshold lives.
