# Structured LLM Candidate-Review Policy

## Scope

The Phase 6D reviewer is a disabled-by-default, second-stage observer for
offers whose Phase 6C agreement route is `LLM_REVIEW`. It receives only the
bounded candidate set already produced by RapidFuzz and scored by LightGBM and
the embedding scorer.

The reviewer may:

1. accept exactly one supplied candidate;
2. reject all supplied candidates; or
3. return uncertain.

It may not create a SKU, request an unrelated Product Master row, alter model
weights, add training examples, update Product Master, or write authoritative
production decisions.

## Default configuration

```yaml
llm_review:
  enabled: false
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
  cache_responses: true
  cache_path: ../data/processed/llm_review_cache.sqlite3
```

Ollama is accessed through its local `/api/generate` endpoint using the Python
standard library. No paid provider or API dependency is required. Provider
construction is lazy; disabled mode performs no network request and creates no
cache.

## Bounded request

Each request contains one offer description, parsed offer attributes, and at
most `maximum_candidates` supplied candidate records. Candidate records carry
only their SKU ID, descriptions, rank, LightGBM probability, embedding
similarity/rank, protein/family/variant/size/pack flags, and controlled
business-rule warnings.

The request never contains unrelated offers or the full Product Master.
Long text fields are deterministically whitespace-normalised and length
bounded. Product text is explicitly treated as untrusted data so it cannot
expand the reviewer's allowed actions.

## Response contract

The provider must return only one JSON object:

```json
{
  "decision": "ACCEPT_CANDIDATE",
  "selected_candidate_id": "SUPPLIED-SKU",
  "confidence": 0.91,
  "reason_codes": ["PROTEIN_MATCH", "FAMILY_MATCH", "SIZE_MATCH"],
  "short_explanation": "The supplied candidate matches the key attributes."
}
```

Allowed decisions are `ACCEPT_CANDIDATE`, `REJECT_ALL`, and `UNCERTAIN`.
Allowed reason codes are:

- `PROTEIN_MATCH`
- `PROTEIN_CONFLICT`
- `FAMILY_MATCH`
- `FAMILY_CONFLICT`
- `SIZE_MATCH`
- `SIZE_CONFLICT`
- `PACK_MATCH`
- `PACK_CONFLICT`
- `VARIANT_MATCH`
- `INSUFFICIENT_INFORMATION`
- `MULTIPLE_PLAUSIBLE_CANDIDATES`
- `NO_VALID_CANDIDATE`

The parser rejects malformed JSON, missing or extra fields, invalid decisions,
unknown or duplicate reason codes, non-finite/out-of-range confidence,
overlong explanations, an invented selected SKU, and a selected SKU on a
non-accept decision.

## Deterministic decision policy

- A valid supplied candidate with confidence at or above the configured
  threshold and no hard deterministic conflict is eligible for
  `LLM_ACCEPT`.
- Low-confidence acceptance routes to `MANUAL_REVIEW`.
- Any protein, strong family, known size/weight, pack-format, mixed-protein,
  feature, or catalogue conflict blocks `LLM_ACCEPT` and routes to
  `MANUAL_REVIEW`.
- `UNCERTAIN` routes to `MANUAL_REVIEW`.
- `REJECT_ALL` uses the explicit `reject_all_route`, which defaults to
  `MANUAL_REVIEW` and may be configured as `NO_MATCH`.
- Invalid output, timeout, provider failure, or disabled mode uses the
  fail route, which is fixed to `MANUAL_REVIEW`.

All Phase 6D routes are diagnostic eligibility only. They do not replace the
Phase 6A or authoritative production decision.

## Provenance and cache

The offer-level result stores provider, model name and cache-isolating model
ID, prompt version, response-schema version, timestamp, canonical request
hash, raw response hash, parsed decision, confidence, selected candidate,
controlled reason codes, short explanation, validation errors, latency,
retry count, cache-hit state, deterministic conflict state, and final route.
The raw response is not written to review CSV artifacts.

The SQLite response cache is keyed by the canonical structured request hash,
exact LLM model ID, prompt version, and schema version. It verifies both the
canonical request content and response SHA-256 before reuse. A response is not
reused across model, prompt, or schema versions. Only schema-valid responses
are cached.

No authentication token, request header, or endpoint credential is stored.
Endpoints containing embedded credentials are rejected.

## Failure handling

Provider and per-response failures are converted to explicit review records
and `MANUAL_REVIEW`. The shadow pipeline remains non-blocking and verifies
that production rows and Product Master remain byte/value unchanged. Human
review and training intake remain separate governed processes.
