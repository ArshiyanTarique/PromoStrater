# Embedding research — archived

**Embeddings are not part of the PromoStrater production pipeline.**

Nothing in this folder runs in production, is imported by `src/sku_mapping/`,
or is exercised by the test suite. It is kept so the experiments do not have to
be repeated, and so a future attempt starts from what was already measured.

The production flow is:

```
preprocess → candidate generation (RapidFuzz, category-gated top-K)
           → feature generation (41 columns)
           → LightGBM → threshold → AUTO_ACCEPT or review
```

No embedding stage exists in it.

## What is here

| File | What it was for |
|---|---|
| `model_vs_embedding_eval.ipynb` | LightGBM vs embedding retrieval comparison |
| `embedding_gpu_evaluation.py` | GPU-backed encoder evaluation |
| `run_embedding_audit.py` | bounded audit harness over embedding retrieval |
| `run_embedding_dry_run.py` | isolated dry run of the embedding scorer |

## What was tested

- **`sku-hashing-384`** (`sklearn-hashing-word-1-2-384-v2`) — the configured
  default. A hashing vectorizer over word 1–2 grams, **not a learned semantic
  model**. It shares RapidFuzz's blind spots: it cannot know that "sambosa"
  means "samosa" or that fries are not nuggets, because both are lexical.
- **`sentence-transformers/all-MiniLM-L6-v2`** — downloaded into a local
  HuggingFace cache for evaluation. Never wired into production; the optional
  `sentence-transformers` dependency exists solely for this path.

## Key findings

**The trained production model never used embeddings.** The registered package
`ranked-v5-cal-20260810T100756Z-matcher` carries 41 features — 19 base, 6
discriminative, 16 group-relative — and **none of them is an embedding,
cosine, or vector feature**. Its `training_config` records
*"group-relative rank features over top-20 RapidFuzz candidates"*. Removing
embeddings therefore required no retraining.

**Embeddings were effectively inert in every real run.** On the 2026-08-12
production run over 254,479 rows the manifest recorded
`embedding_scoring.enabled: false`, `candidates_scored: 0`,
`embedding_retrieval.offers_retrieved: 0`, and `EMBEDDING_UNAVAILABLE` for all
18,001 offers. Runtime attributable to embedding scoring was ~0.0 s.

**Lexical retrieval already covers almost everything.** A K-sweep on real data
measured the shortlist's coverage of rules-supported competitor pairs:

| top-K | coverage |
|---|---|
| 10 | 97.54% |
| **20** | **99.07%** |
| 50 | 99.14% |

K=50 buys 0.07 points for 39% more work. There is very little recall left for a
retrieval layer to recover — which is the strongest single argument against
reintroducing embeddings as a retrieval stage.

**The decision layer's dependence was structural, not empirical.** The old
agreement policy asked "do LightGBM and the embedding agree?", so with
embeddings disabled it returned `EMBEDDING_UNAVAILABLE` and **nothing could
ever auto-accept** — every one of 18,001 offers went to review regardless of
model confidence. That is why the policy was replaced with ML-only threshold
routing, not because embeddings were removed.

## Why embeddings are not used

1. The production model does not consume them.
2. The configured backend is lexical, so it adds cost without adding the
   semantic signal that would justify it.
3. Lexical retrieval already reaches ~99% coverage at K=20.
4. Their only structural role — being the second vote in the agreement policy —
   blocked all auto-acceptance rather than improving it.

## What reintroducing them would require

The pipeline's modular boundaries were deliberately left intact, so an
embedding layer can return without rewriting the ML path:

```
offer → candidate retrieval ─┬─ RapidFuzz (today)
                             └─ optional embedding retrieval
                             ↓
                    merged shortlist → feature generation → LightGBM
```

To do it properly:

1. **Use a real semantic encoder**, not a hashing vectorizer. The hashing
   backend cannot deliver what embeddings are wanted for.
2. **Add it at retrieval, not decision.** Widening the candidate shortlist is
   testable against candidate recall; a second vote in the decision layer is
   what caused the previous failure.
3. **Prove recall gain first.** Measure candidate recall with and without the
   embedding layer on human-labelled offers. The K-sweep above is the baseline
   to beat, and the margin available is under 1 point.
4. **Add embedding-derived features only with retraining.** The current 41-
   feature schema has no slot for them, and the registered package must not be
   altered in place.
5. **Keep it behind a flag** and off by default until the recall measurement
   justifies the runtime.

## Historical data

The learning store still carries `embedding_similarity`, `embedding_status`,
and `embedding_failure_reason` on ~3.3 million rows from runs made while the
feature was wired in. These are historical provenance and were deliberately
**not** dropped. No production code path depends on them.
