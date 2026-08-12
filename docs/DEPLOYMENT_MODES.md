# Deployment Modes

## Status

The repository default is `disabled`. The current registered v3 package is
`SHADOW_MODE_ONLY`, its automatic-production approval flag is false, and its
SHA-256 is:

`f4375c53833a989228d90c8a49aa3ebea9911e8eb6c23519b8183eb4e35d01a1`

The configured `0.85` threshold is user-configured. It is not a production
approval or an 85% accuracy statement.

## Mode comparison

| Property | `disabled` | `shadow` | `assisted` |
| --- | --- | --- | --- |
| Modular inference called | No | Yes | Yes |
| Production-owned rows changed | No | No | Only through explicit final policy |
| Monitoring artifacts | No modular artifacts | Yes | Yes |
| Human questions | No | When persisted and eligible | When persisted and eligible |
| Competitor eligibility from model output | No | No | `AUTO_ACCEPT`/valid `LLM_ACCEPT` only |
| Retraining | Never | Never | Never |

The dashboard deliberately offers only `shadow` and `assisted`; the repository
configuration remains disabled until an operator explicitly starts a run.

## Assisted decision policy

1. No retained candidate: `NO_CANDIDATE`.
2. Model/package unavailable: `MODEL_ERROR` and an ineligible safe fallback.
3. Missing selected master SKU: `MASTER_SKU_NOT_FOUND`.
4. Any deterministic protein, family, known size/weight, pack, mixed-protein,
   feature, or catalogue conflict: `MANUAL_REVIEW`.
5. LightGBM probability at or above `0.85`, same embedding top SKU, valid
   outputs, and no hard conflict: `AUTO_ACCEPT`.
6. Weak agreement or disagreement: structured LLM review when enabled.
7. Valid high-confidence LLM acceptance of a supplied, non-conflicting
   candidate: `LLM_ACCEPT`.
8. Disabled/uncertain/invalid/timed-out/failed/low-confidence LLM output:
   `MANUAL_REVIEW`.

Agreement does not establish correctness. LLM confidence is self-reported and
does not override deterministic conflicts.

## Failure matrix

| Failure | Recorded state | Production behavior |
| --- | --- | --- |
| Embedding unavailable | `EMBEDDING_UNAVAILABLE` | No false agreement; manual/safe fallback |
| LLM unavailable or invalid | Explicit LLM failure status | `MANUAL_REVIEW` |
| Registered package invalid | Package validation failure | `MODEL_ERROR` or existing documented fallback |
| Output validation fails | Run failure; invalid download hidden | No unvalidated artifact offered |
| Learning-store observation fails | Logged non-blocking persistence error | Does not make an ineligible mapping eligible |

## Model activation

Champion–challenger registration and activation are offline control-plane
operations. A passing challenger is registered as assisted-use only and is not
activated automatically. Activation requires the target model, expected
current champion, actor, and reason; rollback uses recorded activation history.

The activation command does not silently rewrite `config/default.yaml`.
Runtime model selection therefore remains an explicit operator action. Do not
select a challenger merely because its files exist; verify registry state,
approval metadata, comparison report, and intended active ID.

Commands:

```powershell
.\.venv\Scripts\python.exe -m sku_mapping.retraining activate-model --help
```

Automatic production matching remains disabled after assisted activation.
