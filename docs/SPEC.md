# ragcache: technical specification

Engineering specification written before and kept in sync with the implementation. Items marked
**REQUIRES USER DECISION** need an input that cannot be derived from the requirements.

## 1. Purpose and boundaries

**Does:** answer questions over a document collection with citations; avoid LLM calls when an equivalent
question was already answered; pick the cheapest adequate model tier for the rest; record cost and latency
per request; expose the numbers.

**Does not:** authenticate end users or enforce per-document permissions (see `permission-aware-rag`);
stream tokens; fine-tune or train models; guarantee answer correctness (it guarantees *citation
validity*, not truth).

**Actors:** *client* (calls `/v1/query`), *operator* (admin API key: ingests, configures, reads metrics),
*LLM provider* (external), *embedding model* (local ONNX).

**Inputs:** documents, queries, runtime config, model registry. **Outputs:** answers with citations, request
log, metrics.

## 2. Requirements

| # | Requirement | Priority | Verified by |
|---|---|---|---|
| F1 | Ingest documents idempotently; unchanged content is a no-op | must | `test_retrieval` |
| F2 | Cache answers by query similarity within (collection, corpus version) | must | `test_cache_router`, `test_pipeline` |
| F3 | Never serve a cached answer past TTL or across corpus versions | must | `test_cache_router` |
| F4 | Guard against numeric / negation mismatches on cache hits | must | `test_cache_router`, `test_api_eval` |
| F5 | Route misses to small or large tier from configurable features | must | `test_cache_router`, `test_pipeline` |
| F6 | Escalate once on invalid answers; fail over on provider errors | must | `test_pipeline` |
| F7 | Never cache abstentions or invalid answers | must | `test_pipeline` |
| F8 | Record tokens, cost, latency, route per request; cost avoided on hits | must | `test_pipeline`, `test_api_eval` |
| F9 | Runtime-tunable config with validation, no redeploy | must | `test_runtime_config`, `test_api_eval` |
| F10 | Dashboard and metrics endpoint | should | `test_api_eval` |
| N1 | No secret stored in the database or logs | must | design; keys referenced by env-var name |
| N2 | Concurrent identical misses cost one LLM call | should | `test_pipeline` |
| N3 | Restart-safe: cache and config survive restart | must | `test_cache_router` |
| N4 | Typed, linted, tested | must | CI |

## 3. Architecture

Modular monolith, one process: `api/app.py` (HTTP) → `pipeline.py` (orchestration) → `cache.py`,
`retrieval.py`, `router.py`, `llm.py`, `models.py` → `db.py` (SQLite, one connection per thread, WAL).

Why not microservices, Redis or a vector database: one process meets the stated requirements, every
component is behind a small class boundary, and each can be replaced (ADR-1, ADR-2).

### Request flow and failure policy

1. Validate input (422), resolve collection (404).
2. Embed the query. Embedding failure → 500, logged as `error`; nothing is fabricated.
3. Cache lookup (skipped if `cache_enabled=false`). On a hit return immediately; log `saved_cost_usd`
   from the original generation.
4. On a miss take the **single-flight** slot for the normalised query. Followers wait, then re-check the
   cache.
5. Retrieve top-k. **No chunks → abstain without calling an LLM.**
6. Route by features; generate on that tier with retry (retryable errors, exponential backoff) then
   failover to the next model in the tier, then the other tier. All failed → 502.
7. Validate: an answer must either be exactly the abstain token or cite only passages that were supplied.
   Invalid on the small tier → one escalation to the large tier (both calls are billed). Invalid after
   that → 502, nothing cached.
8. Cache valid, non-abstaining answers; always write the request row. A failure to write the log never
   fails the request.

## 4. Data model (SQLite, migrations in `src/ragcache/migrations`)

`collections(name, corpus_version)` · `documents(content_hash, UNIQUE(collection, external_id))` ·
`chunks(text, embedding BLOB)` · `cache_entries(collection, corpus_version, embedding, query_signature,
answer, citations_json, source_model, original_cost_usd, expires_at, last_hit_at, hit_count)` ·
`models(tier, provider, model_id, base_url, api_key_env, prices, priority, active)` ·
`requests(...)` (one row per request; query text stored only if `log_query_text`) ·
`runtime_config(key, value_json)` · `schema_migrations`.

Source of truth: documents/chunks are authoritative; `cache_entries` is derived and disposable.
Versioning: forward-only numbered SQL migrations applied at startup inside a transaction.

## 5. Configuration architecture

| Item | Stored | Loaded | Validated | Changed by | Redeploy? |
|---|---|---|---|---|---|
| DB path, embedding model, API keys, timeouts, retries | environment | process start | pydantic-settings | operator | yes |
| Model catalogue and prices | `models` table | per request | CHECK constraints, API schema | CLI / admin API | no |
| Cache, router, retrieval, prompt, logging | `runtime_config` | cached ≤ 5 s | `RuntimeConfig` model | CLI / admin API | no |
| Provider keys | environment, referenced by *name* | at provider construction | presence check | operator | restart or new process |

Genuine constants (SQL schema, the `[n]` citation format, request-id format) stay in code.

## 6. Security and threat model

| Threat | Mitigation | Residual risk |
|---|---|---|
| Admin endpoints exposed | `X-API-Key`, constant-time compare; 503 if the key is unset rather than open | key is a bearer secret: use TLS and a gateway |
| Prompt injection in documents | passages are labelled untrusted; the model has no tools; output validated for citations | a document can still *mislead* an answer; it cannot trigger actions |
| Cache poisoning (a bad answer served to many) | validation before caching; TTL; corpus-version invalidation; abstentions never cached | a valid-looking wrong answer is cached until TTL |
| False cache hits across different intents | threshold 0.95 default, numbers/negation guard, measured precision | measurable false-hit rate (see README) |
| Secret leakage | keys only in env; DB stores env-var names; query text not logged by default | operator error |
| Cross-user data leak via shared cache | **out of scope**: one cache per collection; documented | do not cache permissioned content here |
| DoS via large inputs | query ≤ 4,000 chars, document ≤ 2 MB, body validated | no rate limiting in-process: use a gateway |

## 7. Observability

Structured `requests` table (the system of record for cost/latency), Python logging for provider errors,
retries and unexpected exceptions, `/v1/metrics/summary` and `/dashboard`. Suggested alerts (operator
side): error rate > threshold, escalation rate trending up (router too aggressive toward small),
`saved_cost_usd` flat while traffic grows (cache not effective), p95 generation latency.

## 8. Failure matrix

| Component | Failure | Detection | Behaviour |
|---|---|---|---|
| Embedding model | load/inference error | exception | request 500, `error` row |
| Database | locked/unavailable | exception / `/readyz` 503 | WAL + 30 s busy timeout, then 500; log-write failures swallowed |
| LLM provider | 429/5xx/timeout | `LLMError(retryable)` | retry with backoff, then next model, then other tier, then 502 |
| LLM provider | 4xx/auth/missing key | `LLMError(non-retryable)` | skip to next model; 502 if none left |
| Small model | invalid answer | validator | escalate once |
| Any model | invalid after escalation | validator | 502, not cached |
| Process restart | in-memory matrices lost | n/a | rebuilt lazily from SQLite; single-flight state is per-process |
| Concurrent identical queries | duplicate generation | single-flight | one call, followers read the cache |
| Concurrent ingest during query | stale matrix | corpus_version in key | new version = new namespace |
| Corrupt config write | invalid value | `RuntimeConfig` validation | 422, nothing persisted |

## 9. Testing strategy

Unit and integration tests with test doubles confined to `tests/`; one real-embedding test
(`-m embedding`); an offline evaluation (`eval-threshold`) on a public labelled dataset. **Not covered:**
live provider calls (clients are tested against a local server imitating the wire formats), load/performance tests, multi-process behaviour.

## 10. Deployment

`Dockerfile` (non-root, single container) with a volume for the SQLite file; `ragcache serve --host 0.0.0.0`.
For multiple replicas: Postgres + shared index (not implemented). **REQUIRES USER DECISION:** hosting
target, and whether replicas are needed.

## 11. Architecture decision records

**ADR-1 SQLite + in-memory exact search.** Alternatives: Postgres+pgvector, Redis, Qdrant. Chosen for zero
infrastructure and exactness at the target scale. Consequence: single node; clean seam to replace.

**ADR-2 Modular monolith.** Alternatives: separate cache and router services. Chosen because the request
path is one sequential pipeline and network hops would add latency and failure modes.

**ADR-3 Deterministic router.** Alternatives: learned classifier, LLM-as-router. Chosen because labels do
not exist yet, an LLM router adds cost to the thing meant to save cost, and rules are debuggable. The
escalation log is the future training set.

**ADR-4 Guard + high threshold instead of trusting similarity.** Chosen after measuring precision on a
labelled benchmark; the default was moved from 0.92 to 0.95 as a result.

**ADR-5 Config in the database, secrets in the environment.** Allows runtime tuning without storing
secrets; provider keys are referenced by variable name.

**ADR-6 Thread-local SQLite connections.** A shared connection interleaves transactions across threads;
one connection per thread with WAL avoids that. `:memory:` is rejected because each thread would see a
different database.

## 12. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| Threshold tuned on the wrong distribution | wrong answers served | `eval-threshold` on your own pairs; guard; TTL |
| Router under-escalates | quality loss | escalation rate metric; weights are runtime-tunable |
| Price table drifts from provider pricing | wrong cost figures | prices are data you update; costs stored per request at the time |
| Live provider clients untested against real endpoints | first-run failures | documented; provider interface is small |

## 13. Definition of done and current status

Implemented and tested: F1–F10, N1–N4. **Not implemented / not verified:** (second-stage hit verifier is now implemented: `verifier.py`, `ragcache eval-verifier`),
multi-replica deployment, live LLM integration test, answer-quality evaluation, load testing.
